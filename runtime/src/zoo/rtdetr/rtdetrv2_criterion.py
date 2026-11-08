"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy
import json
from pathlib import Path

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False,
        prototype_contrastive_weight=0.0,
        prototype_temperature=0.1,
        prototype_confusion_margin=0.15,
        confusion_map_path=None,
        event_align_weight=0.0,
        event_query_cls_weight=0.0,
        event_query_bbox_weight=0.0,
        event_query_giou_weight=0.0,
        weight_event_cls_weight=0.0,
        car_margin_weight=0.0,
        car_margin=0.2,
        car_visual_aux_weight=0.0,
        massot_transport_weight=0.0,
        massot_ranking_weight=0.0,
        massot_ranking_margin=0.2,
        cgpr_pair_weight=0.0,
        cgpr_margin_weight=0.0,
        cgpr_margin=0.2,
        cgpr_identity_weight=0.0,
        cfqd_base_vfl_weight=0.0,
        cfqd_parent_weight=0.0,
        cfqd_confusion_weight=0.0,
        cfqd_state_weight=0.0,
        cfqd_state_attention_weight=0.0,
        cfqd_bhcl_weight=0.0,
        cfqd_margin=0.2,
        cfqd_temperature=0.1):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.prototype_contrastive_weight = float(prototype_contrastive_weight)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_confusion_margin = float(prototype_confusion_margin)
        self.event_align_weight = float(event_align_weight)
        self.event_query_cls_weight = float(event_query_cls_weight)
        self.event_query_bbox_weight = float(event_query_bbox_weight)
        self.event_query_giou_weight = float(event_query_giou_weight)
        self.weight_event_cls_weight = float(weight_event_cls_weight)
        self.car_margin_weight = float(car_margin_weight)
        self.car_margin = float(car_margin)
        self.car_visual_aux_weight = float(car_visual_aux_weight)
        self.massot_transport_weight = float(massot_transport_weight)
        self.massot_ranking_weight = float(massot_ranking_weight)
        self.massot_ranking_margin = float(massot_ranking_margin)
        self.cgpr_pair_weight = float(cgpr_pair_weight)
        self.cgpr_margin_weight = float(cgpr_margin_weight)
        self.cgpr_margin = float(cgpr_margin)
        self.cgpr_identity_weight = float(cgpr_identity_weight)
        self.cfqd_base_vfl_weight = float(cfqd_base_vfl_weight)
        self.cfqd_parent_weight = float(cfqd_parent_weight)
        self.cfqd_confusion_weight = float(cfqd_confusion_weight)
        self.cfqd_state_weight = float(cfqd_state_weight)
        self.cfqd_state_attention_weight = float(cfqd_state_attention_weight)
        self.cfqd_bhcl_weight = float(cfqd_bhcl_weight)
        self.cfqd_margin = float(cfqd_margin)
        self.cfqd_temperature = float(cfqd_temperature)
        self.register_buffer(
            'confusion_candidates',
            self._load_confusion_candidates(confusion_map_path, num_classes),
            persistent=False,
        )

    @staticmethod
    def _load_confusion_candidates(confusion_map_path, num_classes):
        if not confusion_map_path:
            return torch.full((num_classes, 1), -1, dtype=torch.long)
        path = Path(confusion_map_path)
        if not path.is_file():
            raise FileNotFoundError(f'Confusion map not found: {path}')
        with path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        mapping = raw.get('confusion_candidates', raw)
        max_candidates = max(
            [len(v) for v in mapping.values() if isinstance(v, list)] or [1]
        )
        candidates = torch.full((num_classes, max_candidates), -1, dtype=torch.long)
        for label, values in mapping.items():
            label = int(label)
            if label < 0 or label >= num_classes or not isinstance(values, list):
                continue
            clean = [
                int(v) for v in values
                if 0 <= int(v) < num_classes and int(v) != label
            ][:max_candidates]
            if clean:
                candidates[label, :len(clean)] = torch.tensor(clean, dtype=torch.long)
        return candidates

    def loss_confusion_prototype(self, outputs, targets, indices):
        if self.prototype_contrastive_weight <= 0:
            return {}
        if 'pred_query_features' not in outputs or 'class_prototypes' not in outputs:
            raise KeyError(
                'Prototype contrastive loss requires pred_query_features and class_prototypes.'
            )

        idx = self._get_src_permutation_idx(indices)
        features = outputs['pred_query_features'][idx]
        labels = torch.cat(
            [t['labels'][target_idx] for t, (_, target_idx) in zip(targets, indices)]
        )
        if features.numel() == 0:
            return {'loss_cpcl': outputs['pred_logits'].float().sum() * 0.0}

        features = F.normalize(features.float(), dim=-1)
        prototypes = F.normalize(outputs['class_prototypes'].float(), dim=-1)
        logits = features @ prototypes.t()

        candidates = self.confusion_candidates.to(labels.device)[labels]
        valid = candidates >= 0
        if valid.any() and self.prototype_confusion_margin > 0:
            rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1)
            rows = rows.expand_as(candidates)[valid]
            cols = candidates[valid]
            logits[rows, cols] = logits[rows, cols] + self.prototype_confusion_margin

        loss = F.cross_entropy(logits / self.prototype_temperature, labels)
        return {'loss_cpcl': loss * self.prototype_contrastive_weight}

    def loss_event_align(self, outputs, targets, indices):
        if self.event_align_weight <= 0:
            return {}
        if 'event_align_logits' not in outputs:
            raise KeyError('event_align_weight > 0 requires event_align_logits in model outputs.')

        idx = self._get_src_permutation_idx(indices)
        logits = outputs['event_align_logits'][idx]
        event_indices = torch.cat(
            [t['event_indices'][target_idx] for t, (_, target_idx) in zip(targets, indices)]
        )
        valid = event_indices >= 0
        if logits.numel() == 0 or not valid.any():
            return {'loss_event_align': outputs['pred_logits'].float().sum() * 0.0}

        logits = logits[valid]
        event_indices = event_indices[valid].to(device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, event_indices)
        return {'loss_event_align': loss * self.event_align_weight}

    def loss_weight_event_cls(self, outputs, targets):
        if self.weight_event_cls_weight <= 0:
            return {}
        if 'weight_event_logits' not in outputs:
            raise KeyError('weight_event_cls_weight > 0 requires weight_event_logits in model outputs.')

        logits = outputs['weight_event_logits']
        pred_logits = []
        target_labels = []
        for batch_idx, target in enumerate(targets):
            if 'event_indices' not in target:
                continue
            event_indices = target['event_indices'].to(device=logits.device, dtype=torch.long)
            valid = (event_indices >= 0) & (event_indices < logits.shape[1])
            if not valid.any():
                continue
            pred_logits.append(logits[batch_idx, event_indices[valid]])
            target_labels.append(target['labels'][valid].to(device=logits.device, dtype=torch.long))

        if not pred_logits:
            return {'loss_weight_event_cls': outputs['pred_logits'].float().sum() * 0.0}

        pred_logits = torch.cat(pred_logits, dim=0)
        target_labels = torch.cat(target_labels, dim=0)
        loss = F.cross_entropy(pred_logits.float(), target_labels)
        return {'loss_weight_event_cls': loss * self.weight_event_cls_weight}

    def loss_event_query(self, outputs, targets):
        if (
            self.event_query_cls_weight <= 0
            and self.event_query_bbox_weight <= 0
            and self.event_query_giou_weight <= 0
        ):
            return {}
        required = ('event_query_logits', 'event_query_boxes', 'event_query_mask')
        if any(k not in outputs for k in required):
            raise KeyError('event query losses require event_query_logits, event_query_boxes, and event_query_mask.')

        logits = outputs['event_query_logits']
        boxes = outputs['event_query_boxes']
        event_mask = outputs['event_query_mask'].to(device=logits.device, dtype=torch.bool)

        pred_logits = []
        pred_boxes = []
        target_labels = []
        target_boxes = []
        for batch_idx, target in enumerate(targets):
            if 'event_indices' not in target:
                continue
            event_indices = target['event_indices'].to(device=logits.device, dtype=torch.long)
            valid = (
                (event_indices >= 0)
                & (event_indices < logits.shape[1])
                & event_mask[batch_idx, event_indices.clamp(min=0, max=max(logits.shape[1] - 1, 0))]
            )
            if not valid.any():
                continue
            event_indices = event_indices[valid]
            pred_logits.append(logits[batch_idx, event_indices])
            pred_boxes.append(boxes[batch_idx, event_indices])
            target_labels.append(target['labels'][valid].to(device=logits.device, dtype=torch.long))
            target_boxes.append(target['boxes'][valid].to(device=logits.device, dtype=boxes.dtype))

        if not pred_logits:
            zero = outputs['pred_logits'].float().sum() * 0.0
            losses = {}
            if self.event_query_cls_weight > 0:
                losses['loss_event_query_cls'] = zero
            if self.event_query_bbox_weight > 0:
                losses['loss_event_query_bbox'] = zero
            if self.event_query_giou_weight > 0:
                losses['loss_event_query_giou'] = zero
            return losses

        pred_logits = torch.cat(pred_logits, dim=0)
        pred_boxes = torch.cat(pred_boxes, dim=0)
        target_labels = torch.cat(target_labels, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)

        losses = {}
        if self.event_query_cls_weight > 0:
            losses['loss_event_query_cls'] = (
                F.cross_entropy(pred_logits.float(), target_labels) * self.event_query_cls_weight
            )
        if self.event_query_bbox_weight > 0:
            losses['loss_event_query_bbox'] = (
                F.l1_loss(pred_boxes, target_boxes, reduction='none').sum() / max(1, target_boxes.shape[0])
            ) * self.event_query_bbox_weight
        if self.event_query_giou_weight > 0:
            giou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            ))
            losses['loss_event_query_giou'] = ((1 - giou).sum() / max(1, target_boxes.shape[0])) * self.event_query_giou_weight
        return losses

    def loss_car(self, outputs, targets, indices, num_boxes):
        losses = {}
        zero = outputs['pred_logits'].float().sum() * 0.0

        if self.car_visual_aux_weight > 0:
            if 'pred_logits_visual' not in outputs:
                raise KeyError('car_visual_aux_weight > 0 requires pred_logits_visual.')
            visual_outputs = {
                'pred_logits': outputs['pred_logits_visual'],
                'pred_boxes': outputs['pred_boxes'],
            }
            visual_loss = self.loss_labels_vfl(
                visual_outputs,
                targets,
                indices,
                num_boxes,
            )['loss_vfl']
            losses['loss_car_visual_aux'] = visual_loss * self.car_visual_aux_weight

        if self.car_margin_weight <= 0:
            return losses
        if 'pred_logits_car' not in outputs or 'car_query_mask' not in outputs:
            raise KeyError('car_margin_weight > 0 requires CAR model outputs.')

        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        labels = torch.cat(
            [target['labels'][target_indices] for target, (_, target_indices) in zip(targets, indices)]
        )
        if query_indices.numel() == 0:
            losses['loss_car_margin'] = zero
            return losses
        selected = outputs['car_query_mask'][batch_indices, query_indices]
        if not selected.any() or outputs['pred_logits_car'].shape[-1] <= 1:
            losses['loss_car_margin'] = zero
            return losses

        logits = outputs['pred_logits_car'][batch_indices, query_indices][selected].float()
        labels = labels[selected].to(device=logits.device, dtype=torch.long)
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = logits.masked_fill(
            F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.bool),
            -torch.inf,
        ).max(dim=1).values
        margin_loss = F.relu(self.car_margin - true_logits + other_logits).mean()
        losses['loss_car_margin'] = margin_loss * self.car_margin_weight
        return losses
    def loss_cgpr(self, outputs, targets, indices):
        losses = {}
        zero = outputs['pred_logits'].float().sum() * 0.0
        if (
            self.cgpr_pair_weight <= 0
            and self.cgpr_margin_weight <= 0
            and self.cgpr_identity_weight <= 0
        ):
            return losses
        required = (
            'pred_logits_cgpr',
            'cgpr_query_mask',
            'cgpr_pair_indices',
            'cgpr_pair_delta',
        )
        if any(key not in outputs for key in required):
            raise KeyError('CGPR losses require CGPR model outputs.')

        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        labels = torch.cat(
            [target['labels'][target_indices] for target, (_, target_indices) in zip(targets, indices)]
        )
        if query_indices.numel() == 0:
            if self.cgpr_pair_weight > 0:
                losses['loss_cgpr_pair'] = zero
            if self.cgpr_margin_weight > 0:
                losses['loss_cgpr_margin'] = zero
            if self.cgpr_identity_weight > 0:
                losses['loss_cgpr_identity'] = zero
            return losses

        selected = outputs['cgpr_query_mask'][batch_indices, query_indices]
        pairs = outputs['cgpr_pair_indices'][batch_indices, query_indices]
        labels = labels.to(device=pairs.device, dtype=torch.long)
        in_pair = selected & ((labels == pairs[:, 0]) | (labels == pairs[:, 1]))
        if in_pair.any():
            current_pairs = pairs[in_pair]
            current_labels = labels[in_pair]
            if 'cgpr_specialist_difference' in outputs:
                difference = outputs['cgpr_specialist_difference'][
                    batch_indices, query_indices
                ][in_pair].float()
            else:
                logits = outputs['pred_logits_cgpr'][
                    batch_indices, query_indices
                ][in_pair].float()
                first = logits.gather(1, current_pairs[:, 0:1]).squeeze(1)
                second = logits.gather(1, current_pairs[:, 1:2]).squeeze(1)
                difference = first - second
            first_is_target = current_labels == current_pairs[:, 0]
            if self.cgpr_pair_weight > 0:
                losses['loss_cgpr_pair'] = F.binary_cross_entropy_with_logits(
                    difference,
                    first_is_target.to(dtype=difference.dtype),
                ) * self.cgpr_pair_weight
            if self.cgpr_margin_weight > 0:
                signed_difference = torch.where(first_is_target, difference, -difference)
                losses['loss_cgpr_margin'] = F.relu(
                    self.cgpr_margin - signed_difference
                ).mean() * self.cgpr_margin_weight
        else:
            if self.cgpr_pair_weight > 0:
                losses['loss_cgpr_pair'] = zero
            if self.cgpr_margin_weight > 0:
                losses['loss_cgpr_margin'] = zero

        if self.cgpr_identity_weight > 0:
            out_of_pair = selected & ~in_pair
            if out_of_pair.any():
                delta = outputs['cgpr_pair_delta'][batch_indices, query_indices][out_of_pair].float()
                losses['loss_cgpr_identity'] = delta.square().mean() * self.cgpr_identity_weight
            else:
                losses['loss_cgpr_identity'] = zero
        return losses

    def loss_cfqd(self, outputs, targets, indices, num_boxes):
        losses = {}
        zero = outputs['pred_logits'].float().sum() * 0.0
        if (
            self.cfqd_base_vfl_weight <= 0
            and self.cfqd_parent_weight <= 0
            and self.cfqd_confusion_weight <= 0
            and self.cfqd_state_weight <= 0
            and self.cfqd_state_attention_weight <= 0
            and self.cfqd_bhcl_weight <= 0
        ):
            return losses

        required = (
            'pred_logits_base',
            'pred_parent_logits',
            'pred_state_logits',
            'pred_cls_features',
            'class_prototypes',
            'cfqd_leaf_to_parent',
            'cfqd_leaf_to_state',
            'cfqd_leaf_balance',
            'cfqd_confusion_candidates',
        )
        if any(key not in outputs for key in required):
            raise KeyError('CFQD losses require CFQD decoder outputs.')

        if self.cfqd_base_vfl_weight > 0:
            base_outputs = {
                'pred_logits': outputs['pred_logits_base'],
                'pred_boxes': outputs['pred_boxes'],
            }
            base_vfl = self.loss_labels_vfl(
                base_outputs, targets, indices, num_boxes
            )['loss_vfl']
            losses['loss_cfqd_base_vfl'] = (
                base_vfl * self.cfqd_base_vfl_weight
            )

        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        labels = torch.cat(
            [
                target['labels'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )
        if query_indices.numel() == 0:
            if self.cfqd_parent_weight > 0:
                losses['loss_cfqd_parent'] = zero
            if self.cfqd_state_weight > 0:
                losses['loss_cfqd_state'] = zero
            if self.cfqd_state_attention_weight > 0:
                losses['loss_cfqd_state_attention'] = zero
            if self.cfqd_confusion_weight > 0:
                losses['loss_cfqd_confusion'] = zero
            if self.cfqd_bhcl_weight > 0:
                losses['loss_cfqd_bhcl'] = zero
            return losses

        logits = outputs['pred_logits'][batch_indices, query_indices].float()
        labels = labels.to(device=logits.device, dtype=torch.long)
        leaf_to_parent = outputs['cfqd_leaf_to_parent'].to(device=logits.device)
        leaf_to_state = outputs['cfqd_leaf_to_state'].to(device=logits.device)

        if self.cfqd_parent_weight > 0:
            parent_logits = outputs['pred_parent_logits'][
                batch_indices, query_indices
            ].float()
            parent_targets = leaf_to_parent.index_select(0, labels)
            losses['loss_cfqd_parent'] = F.cross_entropy(
                parent_logits, parent_targets
            ) * self.cfqd_parent_weight

        if self.cfqd_state_weight > 0:
            state_targets = leaf_to_state.index_select(0, labels)
            valid_states = state_targets >= 0
            if valid_states.any():
                state_logits = outputs['pred_state_logits'][
                    batch_indices, query_indices
                ].float()[valid_states]
                losses['loss_cfqd_state'] = F.cross_entropy(
                    state_logits, state_targets[valid_states]
                ) * self.cfqd_state_weight
            else:
                losses['loss_cfqd_state'] = zero

        if self.cfqd_state_attention_weight > 0:
            if 'pred_state_attention_delta' not in outputs:
                raise KeyError(
                    'cfqd_state_attention_weight requires pred_state_attention_delta.'
                )
            if outputs['pred_state_logits'].shape[-1] != 2:
                raise ValueError('The optional state-attention loss requires binary state logits.')
            state_targets = leaf_to_state.index_select(0, labels)
            valid_states = state_targets >= 0
            if valid_states.any():
                delta = outputs['pred_state_attention_delta'][
                    batch_indices, query_indices
                ].float()[valid_states]
                binary_targets = state_targets[valid_states].float()
                # Pair-constrained residuals are [-delta, +delta], hence 2*delta
                # is the full-versus-empty logit difference.
                losses['loss_cfqd_state_attention'] = (
                    F.binary_cross_entropy_with_logits(
                        2.0 * delta, binary_targets
                    )
                    * self.cfqd_state_attention_weight
                )
            else:
                losses['loss_cfqd_state_attention'] = zero

        confusion_candidates = outputs['cfqd_confusion_candidates'].to(
            device=logits.device
        )
        sample_candidates = confusion_candidates.index_select(0, labels)
        valid_candidates = sample_candidates >= 0
        safe_candidates = sample_candidates.clamp_min(0)

        if self.cfqd_confusion_weight > 0:
            rows_with_candidates = valid_candidates.any(dim=1)
            if rows_with_candidates.any():
                candidate_logits = logits.gather(1, safe_candidates)
                candidate_logits = candidate_logits.masked_fill(
                    ~valid_candidates, -torch.inf
                )
                strongest_confuser = candidate_logits.max(dim=1).values
                target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                margin_loss = F.relu(
                    self.cfqd_margin
                    - target_logits[rows_with_candidates]
                    + strongest_confuser[rows_with_candidates]
                ).mean()
                losses['loss_cfqd_confusion'] = (
                    margin_loss * self.cfqd_confusion_weight
                )
            else:
                losses['loss_cfqd_confusion'] = zero

        if self.cfqd_bhcl_weight > 0:
            features = outputs['pred_cls_features'][
                batch_indices, query_indices
            ].float()
            prototypes = outputs['class_prototypes'].detach().float()
            contrast_logits = F.normalize(features, dim=-1) @ F.normalize(
                prototypes, dim=-1
            ).t()
            contrast_logits = contrast_logits / max(self.cfqd_temperature, 1e-6)
            hard_negative_mask = torch.zeros_like(contrast_logits, dtype=torch.bool)
            candidate_rows = torch.arange(
                labels.numel(), device=logits.device
            ).unsqueeze(1).expand_as(sample_candidates)
            hard_negative_mask[
                candidate_rows[valid_candidates],
                sample_candidates[valid_candidates],
            ] = True
            contrast_logits = contrast_logits + hard_negative_mask.to(
                contrast_logits.dtype
            ) * (self.cfqd_margin / max(self.cfqd_temperature, 1e-6))
            per_sample = F.cross_entropy(
                contrast_logits, labels, reduction='none'
            )
            leaf_balance = outputs['cfqd_leaf_balance'].to(
                device=logits.device, dtype=per_sample.dtype
            )
            per_sample = per_sample * leaf_balance.index_select(0, labels)
            losses['loss_cfqd_bhcl'] = (
                per_sample.mean() * self.cfqd_bhcl_weight
            )
        return losses


    def loss_massot(self, outputs, targets, indices):
        losses = {}
        zero = outputs['pred_logits'].float().sum() * 0.0

        if self.massot_transport_weight > 0:
            required = (
                'transport_plan',
                'massot_query_dustbin',
                'massot_event_mask',
            )
            if any(key not in outputs for key in required):
                raise KeyError('massot_transport_weight > 0 requires MassOT outputs.')
            plan = outputs['transport_plan']
            query_dustbin = outputs['massot_query_dustbin']
            event_mask = outputs['massot_event_mask']
            probabilities = []
            for batch_index, (source_indices, target_indices) in enumerate(indices):
                if source_indices.numel() == 0 or 'event_indices' not in targets[batch_index]:
                    continue
                source_indices = source_indices.to(device=plan.device)
                target_indices = target_indices.to(device=plan.device)
                event_indices = targets[batch_index]['event_indices'][target_indices].long()
                valid = (event_indices >= 0) & (event_indices < plan.shape[-1])
                if valid.any() and plan.shape[-1] > 0:
                    valid = valid & event_mask[
                        batch_index,
                        event_indices.clamp(min=0, max=plan.shape[-1] - 1),
                    ]
                if not valid.any():
                    continue
                source_indices = source_indices[valid]
                event_indices = event_indices[valid]
                numerator = plan[batch_index, source_indices, event_indices]
                denominator = (
                    plan[batch_index, source_indices].sum(dim=-1)
                    + query_dustbin[batch_index, source_indices]
                ).clamp_min(1e-8)
                probabilities.append((numerator / denominator).clamp_min(1e-8))

            if probabilities:
                probability = torch.cat(probabilities)
                losses['loss_massot_transport'] = (
                    -probability.log().mean() * self.massot_transport_weight
                )
            else:
                losses['loss_massot_transport'] = zero

        if self.massot_ranking_weight > 0:
            required = ('transport_energy', 'massot_negative_transport_energy')
            if any(key not in outputs for key in required):
                losses['loss_massot_ranking'] = zero
            else:
                positive = outputs['transport_energy'].float()
                negative = outputs['massot_negative_transport_energy'].float()
                valid = outputs.get('massot_event_mask')
                if valid is not None:
                    valid = valid.any(dim=1)
                else:
                    valid = torch.ones_like(positive, dtype=torch.bool)
                if valid.any():
                    ranking = F.relu(
                        self.massot_ranking_margin + positive[valid] - negative[valid]
                    )
                    losses['loss_massot_ranking'] = (
                        ranking.mean() * self.massot_ranking_weight
                    )
                else:
                    losses['loss_massot_ranking'] = zero
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matching_outputs = outputs_without_aux
        if 'pred_logits_matching' in outputs_without_aux:
            matching_outputs = dict(outputs_without_aux)
            matching_outputs['pred_logits'] = outputs_without_aux['pred_logits_matching']
        matched = self.matcher(matching_outputs, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        losses.update(self.loss_confusion_prototype(outputs, targets, indices))
        losses.update(self.loss_event_align(outputs, targets, indices))
        losses.update(self.loss_event_query(outputs, targets))
        losses.update(self.loss_weight_event_cls(outputs, targets))
        losses.update(self.loss_car(outputs, targets, indices, num_boxes))
        losses.update(self.loss_massot(outputs, targets, indices))
        losses.update(self.loss_cgpr(outputs, targets, indices))
        losses.update(self.loss_cfqd(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        for key, value in outputs.get('training_aux_losses', {}).items():
            if not torch.is_tensor(value):
                raise TypeError(f'training_aux_losses[{key}] must be a tensor.')
            losses[key] = value

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
