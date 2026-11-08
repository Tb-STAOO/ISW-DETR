"""Conservation-verified set refinement for non-spatial mass events."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .mass_count_refinement import MassCountLikelihoodRefinement
from .rtdetr import RTDETR


__all__ = ["ConservationVerifiedSetRefinement", "CVSRRTDETR"]


def _collapse_identity_logits(
    visual_logits, leaf_to_parent, num_parents, reduction="max"
):
    """Broadcast identity evidence to every leaf while erasing state preference."""
    parent_logits = []
    for parent_index in range(int(num_parents)):
        member_indices = (leaf_to_parent == parent_index).nonzero(
            as_tuple=False
        ).flatten()
        values = visual_logits.index_select(-1, member_indices)
        if reduction == "max":
            parent_logits.append(values.amax(dim=-1))
        elif reduction == "logsumexp":
            parent_logits.append(torch.logsumexp(values, dim=-1))
        else:
            raise ValueError("reduction must be 'max' or 'logsumexp'.")
    parent_logits = torch.stack(parent_logits, dim=-1)
    return parent_logits.index_select(-1, leaf_to_parent)


def _load_identity_taxonomy(path, num_classes):
    if not path:
        raise ValueError("identity_taxonomy_path is required when identity_first=True.")
    taxonomy_path = Path(path)
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Identity taxonomy not found: {taxonomy_path}")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
    leaf_to_parent = torch.as_tensor(
        taxonomy.get("leaf_to_parent", []), dtype=torch.long
    )
    if leaf_to_parent.numel() != int(num_classes):
        raise ValueError("Identity taxonomy must map every detector class exactly once.")
    num_parents = int(
        taxonomy.get("num_parents", leaf_to_parent.max().item() + 1)
    )
    if leaf_to_parent.min().item() < 0 or leaf_to_parent.max().item() >= num_parents:
        raise ValueError("Identity taxonomy contains an invalid parent index.")
    return leaf_to_parent, num_parents


class ConservationVerifiedSetRefinement(nn.Module):
    """Use trusted mass events to conservatively refine a small query set."""

    REJECTION_BITS = {
        "missing_records": 1, "numeric_invalid": 2,
        "record_inconsistent": 4, "baseline_conflict": 8,
        "order_invalid": 16, "reference_missing": 32,
        "total_conservation": 64, "endpoint_conservation": 128,
        "chain_conservation": 256, "snapshot_not_independent": 512,
        "likelihood_invalid": 1024,
    }
    # Equality of serialized fields, not an instrument accuracy specification.
    FIELD_TOLERANCE_G = 0.001

    def __init__(
        self,
        num_classes,
        physics_stats_path,
        scene_beta=2.0,
        residual_strength=0.2,
        evidence_clip=6.0,
        candidate_extra=12,
        candidate_iou_threshold=0.9,
        scene_candidate_only=True,
        sensor_sigma_floor_g=0.25,
        total_min_threshold_g=1.0,
        chain_min_threshold_g=0.5,
        threshold_sigma_multiplier=1.0,
        fixed_threshold_mode=False,
        integrity_acceptance_ratio=1.0,
        require_independent_snapshot=False,
        require_positive_mutual_margin=True,
        assignment_strategy="hungarian",
        bypass_integrity_gate=False,
        count_refinement_strength=0.0,
        count_refinement_temperature=1.0,
        ambiguity_adapter_checkpoint=None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.scene_beta = float(scene_beta)
        self.residual_strength = float(residual_strength)
        self.evidence_clip = float(evidence_clip)
        self.candidate_extra = int(candidate_extra)
        self.candidate_iou_threshold = float(candidate_iou_threshold)
        self.scene_candidate_only = bool(scene_candidate_only)
        self.sensor_sigma_floor_g = float(sensor_sigma_floor_g)
        self.total_min_threshold_g = float(total_min_threshold_g)
        self.chain_min_threshold_g = float(chain_min_threshold_g)
        self.threshold_sigma_multiplier = float(threshold_sigma_multiplier)
        self.fixed_threshold_mode = bool(fixed_threshold_mode)
        self.integrity_acceptance_ratio = float(integrity_acceptance_ratio)
        if not 0.0 < self.integrity_acceptance_ratio <= 1.0:
            raise ValueError("integrity_acceptance_ratio must be in (0, 1].")
        self.require_independent_snapshot = bool(require_independent_snapshot)
        self.require_positive_mutual_margin = bool(require_positive_mutual_margin)
        self.assignment_strategy = str(assignment_strategy)
        if self.assignment_strategy not in {"hungarian", "greedy"}:
            raise ValueError("assignment_strategy must be 'hungarian' or 'greedy'.")
        self.bypass_integrity_gate = bool(bypass_integrity_gate)
        count_refinement_strength = float(count_refinement_strength)
        count_refinement_temperature = float(count_refinement_temperature)
        if not math.isfinite(count_refinement_strength) or count_refinement_strength < 0:
            raise ValueError("count_refinement_strength must be finite and nonnegative.")
        if not math.isfinite(count_refinement_temperature) or count_refinement_temperature <= 0:
            raise ValueError("count_refinement_temperature must be finite and positive.")
        # Disabled by default: preserve historical outputs and checkpoint keys.
        self.mass_count_refiner = (
            MassCountLikelihoodRefinement(
                strength=count_refinement_strength,
                temperature=count_refinement_temperature,
                clip=self.evidence_clip,
            )
            if count_refinement_strength > 0
            else None
        )
        self.ambiguity_adapter = None
        self.ambiguity_adapter_inference_scale = 1.0
        if ambiguity_adapter_checkpoint:
            self._load_ambiguity_adapter(ambiguity_adapter_checkpoint)

        stats_path = Path(physics_stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(f"CVSR statistics not found: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        expected = [
            "delta_weight_g",
            "stable_before_g",
            "stable_after_g",
            "stable_std_g",
            "stable_range_g",
            "stable_drift_g",
            "stable_samples",
        ]
        if stats.get("feature_names") != expected:
            raise ValueError(f"CVSR requires feature order {expected}.")
        class_stats = sorted(
            stats.get("class_delta_gaussians", []),
            key=lambda item: int(item["category_id"]),
        )
        if [int(item["category_id"]) for item in class_stats] != list(range(self.num_classes)):
            raise ValueError("CVSR statistics must contain contiguous category ids.")
        if any(not math.isfinite(float(item[key])) for item in class_stats
               for key in ("mean_g", "std_g")) or any(float(item["std_g"]) <= 0 for item in class_stats):
            raise ValueError("Class mass statistics must be finite, with positive SD.")
        self.register_buffer(
            "class_mean_g",
            torch.tensor([abs(float(item["mean_g"])) for item in class_stats]),
        )
        self.register_buffer(
            "class_std_g",
            torch.tensor([max(float(item["std_g"]), 1e-3) for item in class_stats]),
        )

    def _load_ambiguity_adapter(self, checkpoint_path):
        from .ambiguity_rank_adapter import MassGuidedDenoisingRankAdapter

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("class_name") != "MassGuidedDenoisingRankAdapter":
            raise ValueError("Expected a MassGuidedDenoisingRankAdapter checkpoint.")
        selection = payload.get("selection", {})
        if not isinstance(selection, dict) or selection.get("split") != "validation":
            raise ValueError("The ambiguity adapter must have a locked validation selection.")
        if "mode" in selection:
            if selection["mode"] == "no_mass_context":
                raise ValueError("no_mass_context is an offline input ablation, not a native adapter checkpoint.")
            if (self.require_positive_mutual_margin or self.scene_beta != 2.0
                    or self.residual_strength != 0.2):
                raise ValueError(
                    "The selected adapter requires require_positive_mutual_margin=False, "
                    "scene_beta=2.0, and residual_strength=0.2."
                )
        config = payload.get("config")
        if not isinstance(config, dict) or config.get("enabled") is not True:
            raise ValueError("An explicitly enabled adapter configuration is required.")
        scale = float(payload.get("inference_scale", 1.0))
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("The adapter inference scale must be finite and nonnegative.")
        adapter = MassGuidedDenoisingRankAdapter(**config)
        adapter.load_state_dict(payload["state_dict"], strict=True)
        adapter.requires_grad_(False)
        adapter.eval()
        self.ambiguity_adapter = adapter
        self.ambiguity_adapter_inference_scale = scale

    def train(self, mode=True):
        super().train(mode)
        if self.ambiguity_adapter is not None:
            self.ambiguity_adapter.eval()
        return self

    def _event_uncertainty(self, raw_tokens):
        stable_std = raw_tokens[..., 3].abs()
        stable_range = raw_tokens[..., 4].abs()
        stable_drift = raw_tokens[..., 5].abs()
        samples = raw_tokens[..., 6].abs().clamp_min(1.0)
        return torch.sqrt(
            raw_tokens.new_full(stable_std.shape, self.sensor_sigma_floor_g ** 2)
            + (stable_std / samples.sqrt()).square()
            + (stable_range / 4.0).square()
            + stable_drift.square()
        )

    def _event_log_probabilities(self, raw_tokens, event_mask):
        raw_tokens, _, numeric_valid = self._prepare_events(raw_tokens, event_mask)
        delta = raw_tokens[..., 0].abs()
        event_sigma = self._event_uncertainty(raw_tokens)
        total_std = torch.sqrt(
            self.class_std_g.view(1, 1, -1).square()
            + event_sigma.unsqueeze(-1).square()
        ).clamp_min(1e-4)
        z = (delta.unsqueeze(-1) - self.class_mean_g.view(1, 1, -1)) / total_std
        log_likelihood = -0.5 * z.square() - total_std.log()
        log_probability = F.log_softmax(log_likelihood, dim=-1)
        log_probability = torch.where(numeric_valid[:, None, None], log_probability,
                                      torch.full_like(log_probability, -math.log(self.num_classes)))
        return log_probability.masked_fill(~event_mask.unsqueeze(-1), -1e4)

    def _prepare_events(self, raw_tokens, event_mask):
        if raw_tokens.ndim != 3 or raw_tokens.shape[-1] != 7:
            raise ValueError("weight_tokens_raw must have shape [batch, events, 7].")
        if event_mask.shape != raw_tokens.shape[:2]:
            raise ValueError("weight_mask must have shape [batch, events].")
        finite = torch.isfinite(raw_tokens).all(dim=-1)
        statistics_valid = ((raw_tokens[..., 3:5] >= 0).all(dim=-1)
                            & (raw_tokens[..., 6] >= 1)
                            & (raw_tokens[..., 6] == raw_tokens[..., 6].round()))
        numeric_valid = (~event_mask | (finite & statistics_valid)).all(dim=1)
        usable = event_mask & numeric_valid[:, None]
        safe = torch.where(usable.unsqueeze(-1), raw_tokens, torch.zeros_like(raw_tokens))
        return safe, usable, numeric_valid

    def mass_evidence(self, raw_tokens, event_mask, modal_inputs):
        """Shared numeric safety for CVSR and both Gaussian controls."""
        integrity = self._integrity_check(raw_tokens, event_mask, modal_inputs)
        safe, usable, _ = self._prepare_events(raw_tokens, event_mask)
        logp = self._event_log_probabilities(safe, usable)
        finite = (~usable.unsqueeze(-1) | torch.isfinite(logp)).all(dim=(1, 2))
        numeric_valid = integrity["numeric_valid"] & finite
        integrity["rejection_code"] |= (~finite).long() * self.REJECTION_BITS["likelihood_invalid"]
        integrity["numeric_valid"] = numeric_valid
        integrity["gate"] &= numeric_valid
        logp = torch.where(numeric_valid[:, None, None], logp, torch.zeros_like(logp))
        if raw_tokens.shape[1]:
            probability = logp.exp().masked_fill(~usable.unsqueeze(-1), 0.0).max(dim=1).values
        else:
            probability = raw_tokens.new_zeros((raw_tokens.shape[0], self.num_classes))
        probability = torch.where((usable.any(dim=1) & numeric_valid)[:, None], probability,
                                  torch.full_like(probability, 1.0 / self.num_classes))
        evidence = (probability.clamp_min(1e-8).log() + math.log(self.num_classes)).clamp(
            min=-self.evidence_clip, max=self.evidence_clip)
        evidence = torch.where((numeric_valid & event_mask.any(dim=1))[:, None], evidence,
                               torch.zeros_like(evidence))
        return logp, probability, evidence, integrity

    def _integrity_check(self, raw_tokens, event_mask, modal_inputs):
        raw_tokens, _, numeric_valid = self._prepare_events(raw_tokens, event_mask)
        batch_size, max_events, _ = raw_tokens.shape
        device = raw_tokens.device
        scene_total = modal_inputs.get("scene_total_weight_g")
        empty_baseline = modal_inputs.get("empty_baseline_g")
        total_available = modal_inputs.get("scene_total_available")
        total_independent = modal_inputs.get("scene_total_independent")
        event_order = modal_inputs.get("event_order")
        for name in ("scene_total_weight_g", "empty_baseline_g", "scene_total_available", "scene_total_independent"):
            value = modal_inputs.get(name)
            if value is not None and value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch].")
            if value is not None:
                numeric_valid = numeric_valid & torch.isfinite(value.to(device=device)).reshape(batch_size)
        if event_order is not None and event_order.shape != (batch_size, max_events):
            raise ValueError("event_order must have shape [batch, events].")
        if scene_total is None or empty_baseline is None:
            scene_total = raw_tokens.new_full((batch_size,), torch.nan)
            empty_baseline = raw_tokens.new_full((batch_size,), torch.nan)
        else:
            scene_total = scene_total.to(device=device, dtype=torch.float32)
            empty_baseline = empty_baseline.to(device=device, dtype=torch.float32)
        if total_available is None:
            total_available = torch.isfinite(scene_total) & torch.isfinite(empty_baseline)
        else:
            total_available = total_available.to(device=device, dtype=torch.bool)
        if total_independent is None:
            total_independent = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            total_independent = total_independent.to(device=device, dtype=torch.bool)
        if event_order is None:
            event_order = torch.arange(max_events, device=device)[None].expand(batch_size, -1)
        else:
            event_order = event_order.to(device=device)

        reason = torch.zeros(batch_size, dtype=torch.long, device=device)
        reason |= (~event_mask.any(dim=1)).long() * self.REJECTION_BITS["missing_records"]
        reason |= (~numeric_valid).long() * self.REJECTION_BITS["numeric_invalid"]
        reference_valid = total_available & torch.isfinite(scene_total) & torch.isfinite(empty_baseline)
        reason |= (~reference_valid).long() * self.REJECTION_BITS["reference_missing"]
        if self.require_independent_snapshot:
            reason |= (~total_independent).long() * self.REJECTION_BITS["snapshot_not_independent"]
        r_record = raw_tokens.new_full((batch_size,), torch.inf)
        r_baseline = raw_tokens.new_full((batch_size,), torch.inf)
        evaluated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        r_total = raw_tokens.new_full((batch_size,), torch.inf)
        r_end = raw_tokens.new_full((batch_size,), torch.inf)
        r_chain = raw_tokens.new_full((batch_size,), torch.inf)
        chain_pass = torch.zeros(batch_size, dtype=torch.bool, device=device)
        t_total = raw_tokens.new_full((batch_size,), self.total_min_threshold_g)
        t_chain = raw_tokens.new_full((batch_size,), self.chain_min_threshold_g)
        event_sigma = self._event_uncertainty(raw_tokens)
        for batch_index in range(batch_size):
            valid = event_mask[batch_index].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            if not bool(numeric_valid[batch_index]):
                continue
            order_values = event_order[batch_index, valid]
            order_valid = (torch.isfinite(order_values).all()
                           & (order_values == order_values.round()).all()
                           & (order_values.unique().numel() == valid.numel()))
            if not bool(order_valid):
                reason[batch_index] |= self.REJECTION_BITS["order_invalid"]
                continue
            valid = valid[event_order[batch_index, valid].argsort()]
            evaluated[batch_index] = True
            current = raw_tokens[batch_index, valid]
            r_record[batch_index] = (current[:, 0] - (current[:, 2] - current[:, 1])).abs().max()
            r_baseline[batch_index] = (current[0, 1] - empty_baseline[batch_index]).abs()
            if not bool(r_record[batch_index] <= self.FIELD_TOLERANCE_G):
                reason[batch_index] |= self.REJECTION_BITS["record_inconsistent"]
            if reference_valid[batch_index] and not bool(r_baseline[batch_index] <= self.FIELD_TOLERANCE_G):
                reason[batch_index] |= self.REJECTION_BITS["baseline_conflict"]
            sigma = event_sigma[batch_index, valid]
            if not self.fixed_threshold_mode:
                total_sigma = torch.sqrt(sigma[-1].square() + sigma.square().sum())
                t_total[batch_index] = torch.maximum(
                    total_sigma * self.threshold_sigma_multiplier * self.integrity_acceptance_ratio,
                    total_sigma.new_tensor(self.total_min_threshold_g),
                )
            r_total[batch_index] = (
                scene_total[batch_index] - (empty_baseline[batch_index] + current[:, 0].sum())
            ).abs()
            r_end[batch_index] = (scene_total[batch_index] - current[-1, 2]).abs()
            if current.shape[0] <= 1:
                r_chain[batch_index] = 0.0
                chain_pass[batch_index] = True
            else:
                chain_residual = (current[:-1, 2] - current[1:, 1]).abs()
                if self.fixed_threshold_mode:
                    chain_threshold = sigma.new_full(
                        (current.shape[0] - 1,), self.chain_min_threshold_g
                    )
                else:
                    chain_threshold = torch.maximum(
                        torch.sqrt(sigma[:-1].square() + sigma[1:].square())
                        * self.threshold_sigma_multiplier,
                        sigma.new_full((current.shape[0] - 1,), self.chain_min_threshold_g),
                    )
                    chain_threshold = torch.maximum(
                        chain_threshold * self.integrity_acceptance_ratio,
                        sigma.new_full((current.shape[0] - 1,), self.chain_min_threshold_g),
                    )
                r_chain[batch_index] = chain_residual.max()
                t_chain[batch_index] = chain_threshold.max()
                chain_pass[batch_index] = (chain_residual <= chain_threshold).all()
        gate = (
            total_available
            & event_mask.any(dim=1)
            & torch.isfinite(r_total)
            & (r_total <= t_total)
            & (r_end <= t_total)
            & chain_pass
        )
        if self.require_independent_snapshot:
            gate = gate & total_independent
        observable = numeric_valid & event_mask.any(dim=1) & reference_valid & evaluated
        reason |= (observable & ~(r_total <= t_total)).long() * self.REJECTION_BITS["total_conservation"]
        reason |= (observable & ~(r_end <= t_total)).long() * self.REJECTION_BITS["endpoint_conservation"]
        reason |= (observable & ~chain_pass).long() * self.REJECTION_BITS["chain_conservation"]
        gate = gate & (reason == 0)
        if self.bypass_integrity_gate:
            gate = event_mask.any(dim=1) & numeric_valid
        return {
            "gate": gate,
            "numeric_valid": numeric_valid,
            "rejection_code": reason,
            "record_residual_g": r_record,
            "baseline_residual_g": r_baseline,
            "total_residual_g": r_total,
            "end_residual_g": r_end,
            "chain_residual_g": r_chain,
            "total_threshold_g": t_total,
            "chain_threshold_g": t_chain,
            "snapshot_independent": total_independent,
        }

    def _select_candidates(self, logits, boxes, count):
        order = logits.detach().sigmoid().max(dim=-1).values.argsort(descending=True)
        selected = []
        xyxy = box_cxcywh_to_xyxy(boxes.detach().float())
        for query_index in order.tolist():
            if len(selected) >= count:
                break
            if selected:
                reference = torch.tensor(selected, dtype=torch.long, device=boxes.device)
                overlap, _ = box_iou(xyxy[query_index][None], xyxy[reference])
                if float(overlap.max().item()) >= self.candidate_iou_threshold:
                    continue
            selected.append(query_index)
        return torch.tensor(selected, dtype=torch.long, device=boxes.device)

    @staticmethod
    def _mutual_margins(utility, row_indices, column_indices):
        margins = utility.new_empty((len(row_indices),))
        for output_index, (row, column) in enumerate(zip(row_indices, column_indices)):
            value = utility[row, column]
            if utility.shape[1] > 1:
                alternatives = torch.cat([utility[row, :column], utility[row, column + 1 :]])
                row_margin = value - alternatives.max()
            else:
                row_margin = value.new_tensor(torch.inf)
            if utility.shape[0] > 1:
                alternatives = torch.cat([utility[:row, column], utility[row + 1 :, column]])
                column_margin = value - alternatives.max()
            else:
                column_margin = value.new_tensor(torch.inf)
            margins[output_index] = torch.minimum(row_margin, column_margin)
        return margins

    def _assign(self, utility):
        if self.assignment_strategy == "hungarian":
            return linear_sum_assignment(-utility.detach().cpu().numpy())

        row_count, column_count = utility.shape
        ranked = utility.detach().flatten().argsort(descending=True).tolist()
        selected_rows = []
        selected_columns = []
        used_rows = set()
        used_columns = set()
        for flat_index in ranked:
            row = int(flat_index // column_count)
            column = int(flat_index % column_count)
            if row in used_rows or column in used_columns:
                continue
            selected_rows.append(row)
            selected_columns.append(column)
            used_rows.add(row)
            used_columns.add(column)
            if len(selected_rows) >= min(row_count, column_count):
                break
        return np.asarray(selected_rows), np.asarray(selected_columns)

    def _empty_result(self, visual_logits, event_count=0):
        batch_size, query_count, _ = visual_logits.shape
        device = visual_logits.device
        return {
            "pred_logits": visual_logits,
            "pred_logits_scene": visual_logits,
            "event_log_probability": visual_logits.new_zeros(
                (batch_size, event_count, self.num_classes)
            ),
            "integrity_gate": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "integrity_rejection_code": torch.full((batch_size,), self.REJECTION_BITS["missing_records"], dtype=torch.long, device=device),
            "numeric_valid": torch.ones(batch_size, dtype=torch.bool, device=device),
            "integrity_total_residual_g": visual_logits.new_full((batch_size,), torch.inf),
            "integrity_end_residual_g": visual_logits.new_full((batch_size,), torch.inf),
            "integrity_chain_residual_g": visual_logits.new_full((batch_size,), torch.inf),
            "integrity_total_threshold_g": visual_logits.new_full(
                (batch_size,), self.total_min_threshold_g
            ),
            "integrity_chain_threshold_g": visual_logits.new_full(
                (batch_size,), self.chain_min_threshold_g
            ),
            "snapshot_independent": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "candidate_mask": torch.zeros(
                (batch_size, query_count), dtype=torch.bool, device=device
            ),
            "accepted_mask": torch.zeros(
                (batch_size, query_count), dtype=torch.bool, device=device
            ),
            "assigned_event": torch.full(
                (batch_size, query_count), -1, dtype=torch.long, device=device
            ),
            "assignment_margin": visual_logits.new_full(
                (batch_size, query_count), -torch.inf
            ),
        }

    def forward(self, visual_logits, boxes, modal_inputs):
        if visual_logits.ndim != 3 or visual_logits.shape[-1] != self.num_classes:
            raise ValueError("visual_logits must have shape [batch, queries, num_classes].")
        if boxes.shape != (*visual_logits.shape[:2], 4):
            raise ValueError("boxes must have shape [batch, queries, 4].")
        output_dtype = visual_logits.dtype
        with torch.autocast(device_type=visual_logits.device.type, enabled=False):
            result = self._forward_float32(visual_logits.float(), boxes.float(), modal_inputs)
        result["pred_logits"] = result["pred_logits"].to(dtype=output_dtype)
        result["pred_logits_scene"] = result["pred_logits_scene"].to(dtype=output_dtype)
        return result

    def _forward_float32(self, visual_logits, boxes, modal_inputs):
        if modal_inputs is None or modal_inputs.get("weight_tokens_raw") is None:
            return self._empty_result(visual_logits)
        raw_tokens = modal_inputs["weight_tokens_raw"].to(
            device=visual_logits.device, dtype=torch.float32
        )
        event_mask = modal_inputs.get("weight_mask")
        if event_mask is None:
            event_mask = torch.ones(
                raw_tokens.shape[:2], dtype=torch.bool, device=visual_logits.device
            )
        else:
            if not bool((torch.isfinite(event_mask) & ((event_mask == 0) | (event_mask == 1))).all()):
                raise ValueError("weight_mask must contain only boolean or 0/1 values.")
            event_mask = event_mask.to(device=visual_logits.device, dtype=torch.bool)
        if raw_tokens.ndim != 3 or raw_tokens.shape[-1] != 7:
            raise ValueError("weight_tokens_raw must have shape [batch, events, 7].")
        if event_mask.shape != raw_tokens.shape[:2]:
            raise ValueError("weight_mask must have shape [batch, events].")
        if raw_tokens.shape[0] != visual_logits.shape[0]:
            raise ValueError("CVSR modal batch does not match detector batch.")
        if raw_tokens.shape[1] == 0:
            self._integrity_check(raw_tokens, event_mask, modal_inputs)
            return self._empty_result(visual_logits)

        event_log_probability, scene_probability, scene_evidence, integrity = self.mass_evidence(
            raw_tokens, event_mask, modal_inputs)
        gated_evidence = torch.where(integrity["gate"][:, None], scene_evidence,
                                     torch.zeros_like(scene_evidence))
        batch_size, query_count, _ = visual_logits.shape
        candidate_mask = torch.zeros(
            (batch_size, query_count), dtype=torch.bool, device=visual_logits.device
        )
        candidate_lists = []
        for batch_index in range(batch_size):
            if not bool(integrity["gate"][batch_index].item()):
                candidate_lists.append(None)
                continue
            event_count = int(event_mask[batch_index].sum().item())
            candidates = self._select_candidates(
                visual_logits[batch_index],
                boxes[batch_index],
                min(query_count, event_count + self.candidate_extra),
            )
            candidate_mask[batch_index, candidates] = True
            candidate_lists.append(candidates)
        if self.scene_candidate_only:
            scene_gate = candidate_mask.to(gated_evidence.dtype).unsqueeze(-1)
        else:
            scene_gate = integrity["gate"].to(gated_evidence.dtype).view(-1, 1, 1)
        scene_logits = visual_logits + self.scene_beta * gated_evidence.unsqueeze(1) * scene_gate
        refined_logits = scene_logits.clone()
        query_record_residual = (
            torch.zeros_like(refined_logits)
            if (self.ambiguity_adapter is not None
                and self.ambiguity_adapter_inference_scale > 0) else None
        )

        accepted_mask = torch.zeros_like(candidate_mask)
        assigned_event = torch.full(
            (batch_size, query_count), -1, dtype=torch.long, device=visual_logits.device
        )
        assignment_margin = visual_logits.new_full(
            (batch_size, query_count), -torch.inf
        )
        for batch_index in range(batch_size):
            if not bool(integrity["gate"][batch_index].item()):
                continue
            valid_events = event_mask[batch_index].nonzero(as_tuple=False).flatten()
            event_count = int(valid_events.numel())
            candidates = candidate_lists[batch_index]
            if candidates.numel() == 0 or event_count == 0:
                continue
            current_event_log = event_log_probability[batch_index, valid_events]
            scene_log = scene_probability[batch_index].clamp_min(1e-8).log()
            residual = (current_event_log - scene_log.unsqueeze(0)).clamp(
                min=-self.evidence_clip, max=0.0
            )
            utility = torch.logsumexp(
                scene_logits[batch_index, candidates].unsqueeze(1)
                + residual.unsqueeze(0),
                dim=-1,
            )
            rows, columns = self._assign(utility)
            margins = self._mutual_margins(utility, rows, columns)
            for pair_index, (row, column) in enumerate(zip(rows.tolist(), columns.tolist())):
                query_index = int(candidates[row].item())
                event_index = int(valid_events[column].item())
                margin = margins[pair_index]
                assigned_event[batch_index, query_index] = event_index
                assignment_margin[batch_index, query_index] = margin
                if query_record_residual is not None:
                    query_record_residual[batch_index, query_index] = residual[column]
                if self.require_positive_mutual_margin and not bool(margin.item() > 0.0):
                    continue
                accepted_mask[batch_index, query_index] = True
                refined_logits[batch_index, query_index] = (
                    scene_logits[batch_index, query_index]
                    + self.residual_strength * residual[column]
                )

        result = {
            "pred_logits": refined_logits,
            "pred_logits_scene": scene_logits,
            "event_log_probability": event_log_probability,
            "integrity_gate": integrity["gate"],
            "integrity_rejection_code": integrity["rejection_code"],
            "numeric_valid": integrity["numeric_valid"],
            "integrity_record_residual_g": integrity["record_residual_g"],
            "integrity_baseline_residual_g": integrity["baseline_residual_g"],
            "integrity_total_residual_g": integrity["total_residual_g"],
            "integrity_end_residual_g": integrity["end_residual_g"],
            "integrity_chain_residual_g": integrity["chain_residual_g"],
            "integrity_total_threshold_g": integrity["total_threshold_g"],
            "integrity_chain_threshold_g": integrity["chain_threshold_g"],
            "snapshot_independent": integrity["snapshot_independent"],
            "candidate_mask": candidate_mask,
            "accepted_mask": accepted_mask,
            "assigned_event": assigned_event,
            "assignment_margin": assignment_margin,
        }
        if self.ambiguity_adapter is not None:
            if self.ambiguity_adapter_inference_scale == 0:
                # A validation-selected zero scale is the unchanged parent.
                result.update({
                    "rank_adapter_delta": torch.zeros_like(refined_logits),
                    "rank_adapter_features": refined_logits.new_zeros(
                        (*refined_logits.shape, len(self.ambiguity_adapter.FEATURE_NAMES))
                    ),
                    "rank_adapter_applied": torch.zeros_like(integrity["gate"]),
                })
            else:
                adapter_output = self.ambiguity_adapter(
                    visual_logits, refined_logits, scene_evidence,
                    query_record_residual, candidate_mask, integrity["gate"],
                )
                delta = self.ambiguity_adapter_inference_scale * adapter_output["rank_adapter_delta"]
                refined_logits = torch.where(delta == 0, refined_logits, refined_logits + delta)
                result.update(adapter_output)
                result["rank_adapter_delta"] = delta
                result["pred_logits"] = refined_logits
        if self.mass_count_refiner is not None:
            result.update(self.mass_count_refiner(
                refined_logits, event_log_probability, event_mask,
                candidate_mask, integrity["gate"],
            ))
        return result


@register()
class CVSRRTDETR(RTDETR):
    """RT-DETRv2 wrapper with deterministic CVSR classification refinement."""

    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["num_classes"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes=80,
        enable_cvsr=True,
        physics_stats_path=None,
        freeze_detector=True,
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
        cvsr_assignment_strategy="hungarian",
        cvsr_bypass_integrity_gate=False,
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
                assignment_strategy=cvsr_assignment_strategy,
                bypass_integrity_gate=cvsr_bypass_integrity_gate,
                count_refinement_strength=cvsr_count_refinement_strength,
                count_refinement_temperature=cvsr_count_refinement_temperature,
                ambiguity_adapter_checkpoint=cvsr_ambiguity_adapter_checkpoint,
            )

    def _identity_collapsed_logits(self, visual_logits):
        """Remove leaf-state preference while retaining visual identity confidence."""
        if not self.identity_first:
            return visual_logits
        return _collapse_identity_logits(
            visual_logits,
            self.identity_leaf_to_parent,
            self.num_identity_parents,
            self.identity_reduce,
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
        fusion_logits = self._identity_collapsed_logits(visual_logits)
        outputs["pred_logits_visual"] = visual_logits
        outputs["pred_logits_identity"] = fusion_logits
        outputs["pred_logits_matching"] = fusion_logits
        if self.cvsr is None or (self.training and not self.cvsr_apply_during_training):
            outputs["pred_logits"] = fusion_logits
            outputs["pred_logits_cvsr"] = fusion_logits
            return outputs
        cvsr_outputs = self.cvsr(fusion_logits, outputs["pred_boxes"], modal_inputs)
        refined_logits = cvsr_outputs.pop("pred_logits")
        if self.identity_first and self.identity_candidate_only:
            candidate_mask = cvsr_outputs["candidate_mask"].unsqueeze(-1)
            refined_logits = torch.where(candidate_mask, refined_logits, visual_logits)
        if self.identity_first and self.identity_fallback_to_visual:
            gate = cvsr_outputs["integrity_gate"].view(-1, 1, 1)
            refined_logits = torch.where(gate, refined_logits, visual_logits)
        outputs["pred_logits"] = refined_logits
        outputs["pred_logits_cvsr"] = outputs["pred_logits"]
        outputs["pred_logits_scene"] = cvsr_outputs.pop("pred_logits_scene")
        outputs.update({f"cvsr_{key}": value for key, value in cvsr_outputs.items()})
        return outputs
