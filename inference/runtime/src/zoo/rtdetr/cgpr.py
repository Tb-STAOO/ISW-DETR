"""Confusion-graph pairwise reinspection for fine-grained classification."""

from __future__ import annotations

import json
import weakref
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .cvsr import (
    ConservationVerifiedSetRefinement,
    _collapse_identity_logits,
    _load_identity_taxonomy,
)
from .rtdetr import RTDETR


__all__ = ["ConfusionGraphPairwiseReinspection", "CGPRCVSRRTDETR"]


class ConfusionGraphPairwiseReinspection(nn.Module):
    """Apply an antisymmetric residual only to validated confusion pairs."""

    def __init__(
        self,
        backbone,
        num_classes,
        confusion_graph_path,
        hidden_dim=256,
        max_queries=10,
        crop_size=192,
        box_expand_ratio=0.15,
        min_objectness=0.05,
        max_top2_margin=0.3,
        gt_crop_probability=0.5,
        gt_box_jitter=0.15,
        teacher_force_training=True,
        teacher_iou_threshold=0.5,
        edge_specialists=False,
        specialist_blend=0.75,
        specialist_confidence_threshold=0.75,
        specialist_disagreement_only=True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.max_queries = int(max_queries)
        self.crop_size = int(crop_size)
        self.box_expand_ratio = float(box_expand_ratio)
        self.min_objectness = float(min_objectness)
        self.max_top2_margin = float(max_top2_margin)
        self.gt_crop_probability = float(gt_crop_probability)
        self.gt_box_jitter = float(gt_box_jitter)
        self.teacher_force_training = bool(teacher_force_training)
        self.teacher_iou_threshold = float(teacher_iou_threshold)
        self.edge_specialists = bool(edge_specialists)
        self.specialist_blend = float(specialist_blend)
        self.specialist_confidence_threshold = float(
            specialist_confidence_threshold
        )
        self.specialist_disagreement_only = bool(specialist_disagreement_only)
        object.__setattr__(self, "_backbone_ref", weakref.ref(backbone))

        graph_path = Path(confusion_graph_path)
        if not graph_path.is_file():
            raise FileNotFoundError(f"CGPR confusion graph not found: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if int(graph.get("num_classes", -1)) != self.num_classes:
            raise ValueError("CGPR confusion graph class count does not match the model.")
        edge_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.bool)
        edge_id_matrix = torch.full(
            (self.num_classes, self.num_classes), -1, dtype=torch.long
        )
        edge_classes = []
        for edge in graph.get("edges", []):
            classes = edge.get("classes", edge)
            if len(classes) != 2:
                continue
            first, second = (int(value) for value in classes)
            if 0 <= first < self.num_classes and 0 <= second < self.num_classes:
                first, second = sorted((first, second))
                if edge_matrix[first, second]:
                    continue
                edge_id = len(edge_classes)
                edge_matrix[first, second] = True
                edge_matrix[second, first] = True
                edge_id_matrix[first, second] = edge_id
                edge_id_matrix[second, first] = edge_id
                edge_classes.append([first, second])
        if not edge_classes:
            raise ValueError("CGPR confusion graph does not contain valid edges.")
        self.register_buffer("edge_matrix", edge_matrix, persistent=True)
        self.register_buffer("edge_id_matrix", edge_id_matrix, persistent=True)
        self.register_buffer(
            "edge_classes",
            torch.tensor(edge_classes, dtype=torch.long),
            persistent=True,
        )

        backbone_channels = int(backbone.out_channels[-1])
        self.specialist_norm = None
        self.specialist_head = None
        if self.edge_specialists:
            self.specialist_norm = nn.LayerNorm(backbone_channels)
            self.specialist_head = nn.Linear(backbone_channels, len(edge_classes))
            nn.init.zeros_(self.specialist_head.weight)
            nn.init.zeros_(self.specialist_head.bias)
        self.detail_projection = nn.Sequential(
            nn.Linear(backbone_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.prototype_norm = nn.LayerNorm(hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        if self.edge_specialists:
            for legacy_module in (
                self.detail_projection,
                self.query_norm,
                self.prototype_norm,
                self.pair_head,
            ):
                legacy_module.requires_grad_(False)

    @property
    def shared_backbone(self):
        backbone = self._backbone_ref()
        if backbone is None:
            raise RuntimeError("CGPR shared backbone is no longer available.")
        return backbone

    def _zero_anchor(self, reference):
        if not self.training:
            return reference.new_zeros(())
        return sum(
            (parameter.reshape(-1)[0].float() * 0.0 for parameter in self.parameters()),
            start=reference.new_zeros((), dtype=torch.float32),
        ).to(dtype=reference.dtype)

    def _select_queries(self, logits):
        probabilities = logits.detach().sigmoid()
        top_values, top_indices = probabilities.topk(2, dim=-1)
        margin = top_values[..., 0] - top_values[..., 1]
        is_edge = self.edge_matrix[top_indices[..., 0], top_indices[..., 1]]
        valid = (
            is_edge
            & (top_values[..., 0] >= self.min_objectness)
            & (margin <= self.max_top2_margin)
        )
        ambiguity = top_values[..., 0] * (1.0 - margin)
        selected_batch, selected_query = [], []
        for batch_index in range(logits.shape[0]):
            indices = valid[batch_index].nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            count = min(self.max_queries, int(indices.numel()))
            indices = indices[ambiguity[batch_index, indices].topk(count).indices]
            selected_batch.append(torch.full_like(indices, batch_index))
            selected_query.append(indices)
        if not selected_batch:
            empty = torch.zeros(0, dtype=torch.long, device=logits.device)
            return empty, empty, top_indices
        return torch.cat(selected_batch), torch.cat(selected_query), top_indices
    def _select_training_queries(
        self,
        logits,
        boxes,
        targets,
        base_batch_indices,
        base_query_indices,
        top_indices,
    ):
        """Add localized confusion-pair supervision without changing inference."""
        if not self.training or not self.teacher_force_training or targets is None:
            pairs = top_indices[base_batch_indices, base_query_indices]
            return base_batch_indices, base_query_indices, pairs

        probabilities = logits.detach().sigmoid()
        selected_batch, selected_query, selected_pairs = [], [], []
        for batch_index in range(logits.shape[0]):
            candidates = {}
            target_boxes = targets[batch_index].get("boxes")
            target_labels = targets[batch_index].get("labels")
            if target_boxes is not None and target_labels is not None and target_boxes.numel() > 0:
                overlap, _ = box_iou(
                    box_cxcywh_to_xyxy(boxes[batch_index].detach().float()),
                    box_cxcywh_to_xyxy(target_boxes.detach().float()),
                )
                best_overlap, best_queries = overlap.max(dim=0)
                for target_index in range(target_labels.numel()):
                    localization = float(best_overlap[target_index].item())
                    if localization < self.teacher_iou_threshold:
                        continue
                    query_index = int(best_queries[target_index].item())
                    true_class = int(target_labels[target_index].item())
                    predicted_class = int(top_indices[batch_index, query_index, 0].item())
                    if predicted_class == true_class:
                        neighbors = self.edge_matrix[true_class].nonzero(as_tuple=False).flatten()
                        if neighbors.numel() == 0:
                            continue
                        neighbor_scores = probabilities[batch_index, query_index, neighbors]
                        confused_class = int(neighbors[neighbor_scores.argmax()].item())
                        pair = (true_class, confused_class)
                        wrong_class_priority = 0
                    elif bool(self.edge_matrix[predicted_class, true_class].item()):
                        pair = (predicted_class, true_class)
                        wrong_class_priority = 1
                    else:
                        continue
                    pair_margin = abs(
                        float(probabilities[batch_index, query_index, pair[0]].item())
                        - float(probabilities[batch_index, query_index, pair[1]].item())
                    )
                    priority = (2, wrong_class_priority, -pair_margin, localization)
                    existing = candidates.get(query_index)
                    if existing is None or priority > existing[0]:
                        candidates[query_index] = (priority, pair)

            base_queries = base_query_indices[base_batch_indices == batch_index].tolist()
            for query_index in base_queries:
                query_index = int(query_index)
                if query_index in candidates:
                    continue
                pair = tuple(int(value) for value in top_indices[batch_index, query_index].tolist())
                ambiguity = float(
                    probabilities[batch_index, query_index, pair[0]].item()
                    - probabilities[batch_index, query_index, pair[1]].item()
                )
                candidates[query_index] = ((1, 0, -ambiguity, 0.0), pair)

            ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[: self.max_queries]
            for query_index, (_, pair) in ranked:
                selected_batch.append(batch_index)
                selected_query.append(query_index)
                selected_pairs.append(pair)

        if not selected_batch:
            empty = torch.zeros(0, dtype=torch.long, device=logits.device)
            return empty, empty, empty.reshape(0, 2)
        return (
            torch.tensor(selected_batch, dtype=torch.long, device=logits.device),
            torch.tensor(selected_query, dtype=torch.long, device=logits.device),
            torch.tensor(selected_pairs, dtype=torch.long, device=logits.device),
        )

    def _training_crop_boxes(self, boxes, batch_indices, query_indices, targets):
        selected = boxes[batch_indices, query_indices].detach().float().clone()
        if not self.training or targets is None or self.gt_crop_probability <= 0:
            return selected
        for selected_index, (batch_index, query_index) in enumerate(
            zip(batch_indices.tolist(), query_indices.tolist())
        ):
            if torch.rand((), device=boxes.device) >= self.gt_crop_probability:
                continue
            target_boxes = targets[batch_index].get("boxes")
            if target_boxes is None or target_boxes.numel() == 0:
                continue
            overlap, _ = box_iou(
                box_cxcywh_to_xyxy(boxes[batch_index, query_index].detach().float()[None]),
                box_cxcywh_to_xyxy(target_boxes.detach().float()),
            )
            best_overlap, best_index = overlap[0].max(dim=0)
            if float(best_overlap.item()) <= 0.0:
                continue
            crop_box = target_boxes[best_index].detach().float().clone()
            jitter = self.gt_box_jitter
            crop_box[:2] += (torch.rand(2, device=boxes.device) * 2.0 - 1.0) * jitter * crop_box[2:]
            crop_box[2:] *= 1.0 + (torch.rand(2, device=boxes.device) * 2.0 - 1.0) * jitter
            crop_box[:2].clamp_(0.0, 1.0)
            crop_box[2:].clamp_(1e-4, 1.0)
            selected[selected_index] = crop_box
        return selected

    def _build_rois(self, selected, batch_indices, detail_sizes):
        center_x, center_y, width, height = selected.unbind(dim=-1)
        width = width * (1.0 + self.box_expand_ratio)
        height = height * (1.0 + self.box_expand_ratio)
        image_height = detail_sizes[batch_indices, 0].float().clamp_min(1.0)
        image_width = detail_sizes[batch_indices, 1].float().clamp_min(1.0)
        x1 = ((center_x - width / 2.0) * image_width).clamp_min(0.0)
        y1 = ((center_y - height / 2.0) * image_height).clamp_min(0.0)
        x2 = torch.minimum((center_x + width / 2.0) * image_width, image_width)
        y2 = torch.minimum((center_y + height / 2.0) * image_height, image_height)
        x1 = torch.minimum(x1, (image_width - 1.0).clamp_min(0.0))
        y1 = torch.minimum(y1, (image_height - 1.0).clamp_min(0.0))
        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)
        return torch.stack([batch_indices.float(), x1, y1, x2, y2], dim=-1)

    def forward(
        self,
        logits,
        boxes,
        query_features,
        class_prototypes,
        visual_inputs,
        targets=None,
    ):
        batch_size, query_count, _ = logits.shape
        query_mask = torch.zeros(
            (batch_size, query_count), dtype=torch.bool, device=logits.device
        )
        pair_indices = torch.full(
            (batch_size, query_count, 2), -1, dtype=torch.long, device=logits.device
        )
        pair_delta = logits.new_zeros((batch_size, query_count))
        specialist_difference = logits.new_zeros((batch_size, query_count))
        if visual_inputs is None or visual_inputs.get("detail_images") is None:
            return (
                logits + self._zero_anchor(logits),
                query_mask,
                pair_indices,
                pair_delta,
                specialist_difference,
            )

        detail_images = visual_inputs["detail_images"]
        detail_sizes = visual_inputs["detail_sizes"]
        batch_indices, query_indices, all_top_indices = self._select_queries(logits)
        batch_indices, query_indices, pairs = self._select_training_queries(
            logits, boxes, targets, batch_indices, query_indices, all_top_indices
        )
        if batch_indices.numel() > 0:
            has_detail = (detail_sizes[batch_indices] > 0).all(dim=-1)
            batch_indices, query_indices = batch_indices[has_detail], query_indices[has_detail]
            pairs = pairs[has_detail]
        if batch_indices.numel() == 0:
            return (
                logits + self._zero_anchor(logits),
                query_mask,
                pair_indices,
                pair_delta,
                specialist_difference,
            )

        selected_boxes = self._training_crop_boxes(
            boxes, batch_indices, query_indices, targets
        )
        rois = self._build_rois(selected_boxes, batch_indices, detail_sizes)
        crops = roi_align(
            detail_images,
            rois,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        backbone = self.shared_backbone
        with torch.no_grad():
            backbone.eval()
            detail_features = backbone(crops)[-1].mean(dim=(-2, -1)).float()
        if self.edge_specialists:
            edge_ids = self.edge_id_matrix[pairs[:, 0], pairs[:, 1]]
            valid_edges = edge_ids >= 0
            all_scores = self.specialist_head(self.specialist_norm(detail_features))
            # 修改下面这行：指定 dtype 为 all_scores.dtype
            canonical_scores = detail_features.new_zeros((detail_features.shape[0],), dtype=all_scores.dtype)
            canonical_scores[valid_edges] = all_scores[
                valid_edges, edge_ids[valid_edges]
            ]
            canonical_first = self.edge_classes[
                edge_ids.clamp_min(0), 0
            ]
            signed_scores = torch.where(
                pairs[:, 0] == canonical_first,
                canonical_scores,
                -canonical_scores,
            )
            visual_difference = (
                logits[batch_indices, query_indices, pairs[:, 0]]
                - logits[batch_indices, query_indices, pairs[:, 1]]
            ).float()
            apply_specialist = valid_edges
            if not self.training:
                confidence = torch.sigmoid(signed_scores.abs())
                apply_specialist = apply_specialist & (
                    confidence >= self.specialist_confidence_threshold
                )
                if self.specialist_disagreement_only:
                    apply_specialist = apply_specialist & (
                        signed_scores * visual_difference < 0
                    )
            delta = 0.5 * self.specialist_blend * signed_scores
            delta = torch.where(apply_specialist, delta, torch.zeros_like(delta))
        else:
            detail_features = self.detail_projection(detail_features)
            prototype_difference = (
                class_prototypes[pairs[:, 0]] - class_prototypes[pairs[:, 1]]
            ).detach()
            selected_queries = query_features[batch_indices, query_indices].detach()
            delta = self.pair_head(
                torch.cat(
                    [
                        self.query_norm(selected_queries),
                        detail_features,
                        self.prototype_norm(prototype_difference),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            signed_scores = delta
            apply_specialist = torch.ones_like(delta, dtype=torch.bool)

        residual = logits.new_zeros(logits.shape)
        residual[batch_indices, query_indices, pairs[:, 0]] = delta.to(logits.dtype)
        residual[batch_indices, query_indices, pairs[:, 1]] = -delta.to(logits.dtype)
        applied_batch = batch_indices[apply_specialist]
        applied_query = query_indices[apply_specialist]
        query_mask[applied_batch, applied_query] = True
        pair_indices[applied_batch, applied_query] = pairs[apply_specialist]
        pair_delta[batch_indices, query_indices] = delta.to(logits.dtype)
        specialist_difference[batch_indices, query_indices] = signed_scores.to(
            logits.dtype
        )
        return (
            logits + residual,
            query_mask,
            pair_indices,
            pair_delta,
            specialist_difference,
        )


@register()
class CGPRCVSRRTDETR(RTDETR):
    """Frozen RT-DETRv2 with optional CGPR and deterministic CVSR refinement."""

    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["num_classes"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes=80,
        enable_cgpr=True,
        enable_cvsr=False,
        confusion_graph_path=None,
        physics_stats_path=None,
        freeze_detector=True,
        cgpr_max_queries=10,
        cgpr_crop_size=192,
        cgpr_box_expand_ratio=0.15,
        cgpr_min_objectness=0.05,
        cgpr_max_top2_margin=0.3,
        cgpr_gt_crop_probability=0.5,
        cgpr_gt_box_jitter=0.15,
        cgpr_teacher_force_training=True,
        cgpr_teacher_iou_threshold=0.5,
        cgpr_edge_specialists=False,
        cgpr_specialist_blend=0.75,
        cgpr_specialist_confidence_threshold=0.75,
        cgpr_specialist_disagreement_only=True,
        cvsr_apply_during_training=False,
        cvsr_scene_beta=2.0,
        cvsr_residual_strength=0.2,
        cvsr_evidence_clip=6.0,
        cvsr_candidate_extra=12,
        cvsr_candidate_iou_threshold=0.9,
        cvsr_scene_candidate_only=True,
        cvsr_sensor_sigma_floor_g=0.25,
        cvsr_total_min_threshold_g=1.0,
        cvsr_chain_min_threshold_g=0.5,
        cvsr_threshold_sigma_multiplier=1.0,
        cvsr_fixed_threshold_mode=False,
        cvsr_integrity_acceptance_ratio=0.2,
        cvsr_require_independent_snapshot=False,
        cvsr_require_positive_mutual_margin=True,
        identity_first=False,
        identity_taxonomy_path=None,
        identity_reduce="max",
        identity_fallback_to_visual=True,
        identity_candidate_only=True,
        cvsr_count_refinement_strength=0.0,
        cvsr_count_refinement_temperature=1.0,
        cvsr_ambiguity_adapter_checkpoint=None,
    ):
        super().__init__(backbone=backbone, encoder=encoder, decoder=decoder)
        self.freeze_detector = bool(freeze_detector)
        self.cvsr_apply_during_training = bool(cvsr_apply_during_training)
        self.identity_first = bool(identity_first)
        self.identity_reduce = str(identity_reduce)
        self.identity_fallback_to_visual = bool(identity_fallback_to_visual)
        self.identity_candidate_only = bool(identity_candidate_only)
        if self.identity_reduce not in {"max", "logsumexp"}:
            raise ValueError("identity_reduce must be 'max' or 'logsumexp'.")
        if self.identity_first:
            leaf_to_parent, num_parents = _load_identity_taxonomy(
                identity_taxonomy_path, num_classes
            )
        else:
            leaf_to_parent = torch.arange(int(num_classes), dtype=torch.long)
            num_parents = int(num_classes)
        self.num_identity_parents = num_parents
        self.register_buffer(
            "identity_leaf_to_parent", leaf_to_parent, persistent=False
        )
        if self.freeze_detector:
            self.backbone.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
        hidden_dim = int(getattr(decoder, "hidden_dim", 256))
        self.cgpr = None
        if enable_cgpr:
            if not confusion_graph_path:
                raise ValueError("confusion_graph_path is required when enable_cgpr=True.")
            self.cgpr = ConfusionGraphPairwiseReinspection(
                backbone=backbone,
                num_classes=num_classes,
                confusion_graph_path=confusion_graph_path,
                hidden_dim=hidden_dim,
                max_queries=cgpr_max_queries,
                crop_size=cgpr_crop_size,
                box_expand_ratio=cgpr_box_expand_ratio,
                min_objectness=cgpr_min_objectness,
                max_top2_margin=cgpr_max_top2_margin,
                gt_crop_probability=cgpr_gt_crop_probability,
                gt_box_jitter=cgpr_gt_box_jitter,
                teacher_force_training=cgpr_teacher_force_training,
                teacher_iou_threshold=cgpr_teacher_iou_threshold,
                edge_specialists=cgpr_edge_specialists,
                specialist_blend=cgpr_specialist_blend,
                specialist_confidence_threshold=cgpr_specialist_confidence_threshold,
                specialist_disagreement_only=cgpr_specialist_disagreement_only,
            )
        self.cvsr = None
        if enable_cvsr:
            if not physics_stats_path:
                raise ValueError("physics_stats_path is required when enable_cvsr=True.")
            self.cvsr = ConservationVerifiedSetRefinement(
                num_classes=num_classes,
                physics_stats_path=physics_stats_path,
                scene_beta=cvsr_scene_beta,
                residual_strength=cvsr_residual_strength,
                evidence_clip=cvsr_evidence_clip,
                candidate_extra=cvsr_candidate_extra,
                candidate_iou_threshold=cvsr_candidate_iou_threshold,
                scene_candidate_only=cvsr_scene_candidate_only,
                sensor_sigma_floor_g=cvsr_sensor_sigma_floor_g,
                total_min_threshold_g=cvsr_total_min_threshold_g,
                chain_min_threshold_g=cvsr_chain_min_threshold_g,
                threshold_sigma_multiplier=cvsr_threshold_sigma_multiplier,
                fixed_threshold_mode=cvsr_fixed_threshold_mode,
                integrity_acceptance_ratio=cvsr_integrity_acceptance_ratio,
                require_independent_snapshot=cvsr_require_independent_snapshot,
                require_positive_mutual_margin=cvsr_require_positive_mutual_margin,
                count_refinement_strength=cvsr_count_refinement_strength,
                count_refinement_temperature=cvsr_count_refinement_temperature,
                ambiguity_adapter_checkpoint=cvsr_ambiguity_adapter_checkpoint,
            )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_detector:
            self.backbone.eval()
            self.encoder.eval()
            self.decoder.eval()
        return self

    def forward(self, x, targets=None, modal_inputs=None, visual_inputs=None):
        features = self.encoder(self.backbone(x))
        outputs = self.decoder(features, targets=targets)
        visual_logits = outputs["pred_logits"]
        outputs["pred_logits_visual"] = visual_logits
        outputs["pred_logits_matching"] = visual_logits

        cgpr_logits = visual_logits
        if self.cgpr is not None:
            if "pred_query_features" not in outputs or "class_prototypes" not in outputs:
                raise KeyError("CGPR requires decoder expose_query_features=True.")
            cgpr_logits, mask, pairs, delta, specialist_difference = self.cgpr(
                visual_logits,
                outputs["pred_boxes"],
                outputs["pred_query_features"],
                outputs["class_prototypes"],
                visual_inputs,
                targets=targets,
            )
            outputs["cgpr_query_mask"] = mask
            outputs["cgpr_pair_indices"] = pairs
            outputs["cgpr_pair_delta"] = delta
            outputs["cgpr_specialist_difference"] = specialist_difference
        outputs["pred_logits_cgpr"] = cgpr_logits

        identity_logits = (
            _collapse_identity_logits(
                cgpr_logits,
                self.identity_leaf_to_parent,
                self.num_identity_parents,
                self.identity_reduce,
            )
            if self.identity_first
            else cgpr_logits
        )
        outputs["pred_logits_identity"] = identity_logits
        outputs["pred_logits_matching"] = identity_logits

        if self.cvsr is None or (self.training and not self.cvsr_apply_during_training):
            outputs["pred_logits"] = identity_logits
            outputs["pred_logits_cvsr"] = identity_logits
            return outputs
        cvsr_outputs = self.cvsr(identity_logits, outputs["pred_boxes"], modal_inputs)
        refined_logits = cvsr_outputs.pop("pred_logits")
        if self.identity_first and self.identity_candidate_only:
            candidate_mask = cvsr_outputs["candidate_mask"].unsqueeze(-1)
            refined_logits = torch.where(candidate_mask, refined_logits, cgpr_logits)
        if self.identity_first and self.identity_fallback_to_visual:
            gate = cvsr_outputs["integrity_gate"].view(-1, 1, 1)
            refined_logits = torch.where(gate, refined_logits, cgpr_logits)
        outputs["pred_logits"] = refined_logits
        outputs["pred_logits_cvsr"] = outputs["pred_logits"]
        outputs["pred_logits_scene"] = cvsr_outputs.pop("pred_logits_scene")
        outputs.update({f"cvsr_{key}": value for key, value in cvsr_outputs.items()})
        return outputs
