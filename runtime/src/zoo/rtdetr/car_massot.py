"""Confusion-aware visual reinspection and mass-conserving event fusion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ...core import register
from .rtdetr import RTDETR


__all__ = [
    'ConfusionAwareReinspection',
    'MassConservingOptimalTransport',
    'CARMassOTRTDETR',
]


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class ConfusionAwareReinspection(nn.Module):
    """Reinspect uncertain detector queries in aligned high-resolution crops."""

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nhead=8,
        train_max_queries=30,
        eval_max_queries=10,
        candidate_topk=3,
        use_difference_queries=True,
        crop_size=160,
        box_expand_ratio=0.15,
        min_objectness=0.05,
        max_top2_margin=0.35,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.train_max_queries = int(train_max_queries)
        self.eval_max_queries = int(eval_max_queries)
        self.candidate_topk = min(int(candidate_topk), self.num_classes)
        self.use_difference_queries = bool(use_difference_queries)
        self.crop_size = int(crop_size)
        self.box_expand_ratio = float(box_expand_ratio)
        self.min_objectness = float(min_objectness)
        self.max_top2_margin = float(max_top2_margin)

        self.detail_encoder = nn.Sequential(
            _ConvNormAct(3, 32),
            _ConvNormAct(32, 64),
            _ConvNormAct(64, 128),
            _ConvNormAct(128, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.detail_norm = nn.LayerNorm(hidden_dim)
        self.prototype_norm = nn.LayerNorm(hidden_dim)
        self.detail_attn = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _zero_gradient_anchor(self, reference):
        if not self.training:
            return reference.new_zeros(())
        anchor = None
        for parameter in self.parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            term = parameter.reshape(-1)[0].float() * 0.0
            anchor = term if anchor is None else anchor + term
        if anchor is None:
            return reference.new_zeros(())
        return anchor.to(device=reference.device, dtype=reference.dtype)

    def _select_queries(self, logits):
        probabilities = logits.detach().sigmoid()
        objectness = probabilities.max(dim=-1).values
        if self.num_classes > 1:
            top2 = probabilities.topk(k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
        else:
            margin = torch.ones_like(objectness)
        distribution = torch.softmax(logits.detach().float(), dim=-1).clamp_min(1e-8)
        entropy = -(distribution * distribution.log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(self.num_classes, 2)), 1e-6)
        ambiguity = objectness * (entropy + 1.0 - margin)
        valid = (objectness >= self.min_objectness) & (margin <= self.max_top2_margin)

        limit = self.train_max_queries if self.training else self.eval_max_queries
        selected_batch = []
        selected_query = []
        for batch_index in range(logits.shape[0]):
            indices = valid[batch_index].nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            count = min(limit, int(indices.numel()))
            scores = ambiguity[batch_index, indices]
            indices = indices[scores.topk(count).indices]
            selected_batch.append(torch.full_like(indices, batch_index))
            selected_query.append(indices)

        if not selected_batch:
            empty = torch.zeros(0, dtype=torch.long, device=logits.device)
            return empty, empty
        return torch.cat(selected_batch), torch.cat(selected_query)

    def _build_rois(self, boxes, batch_indices, query_indices, detail_sizes):
        selected = boxes[batch_indices, query_indices].detach().float()
        center_x, center_y, width, height = selected.unbind(dim=-1)
        scale = 1.0 + self.box_expand_ratio
        width = width * scale
        height = height * scale

        image_height = detail_sizes[batch_indices, 0].float().clamp_min(1.0)
        image_width = detail_sizes[batch_indices, 1].float().clamp_min(1.0)
        x1 = torch.minimum(
            ((center_x - 0.5 * width) * image_width).clamp_min(0.0),
            (image_width - 1.0).clamp_min(0.0),
        )
        y1 = torch.minimum(
            ((center_y - 0.5 * height) * image_height).clamp_min(0.0),
            (image_height - 1.0).clamp_min(0.0),
        )
        x2 = torch.minimum((center_x + 0.5 * width) * image_width, image_width)
        y2 = torch.minimum((center_y + 0.5 * height) * image_height, image_height)
        x2 = torch.maximum(x2, x1 + 1.0)
        y2 = torch.maximum(y2, y1 + 1.0)
        return torch.stack([batch_indices.float(), x1, y1, x2, y2], dim=-1)

    def forward(self, logits, boxes, query_features, class_prototypes, visual_inputs):
        batch_size, query_count, class_count = logits.shape
        selected_mask = torch.zeros(
            (batch_size, query_count),
            dtype=torch.bool,
            device=logits.device,
        )
        empty_candidates = torch.full(
            (batch_size, query_count, self.candidate_topk),
            -1,
            dtype=torch.long,
            device=logits.device,
        )
        empty_residuals = logits.new_zeros(
            (batch_size, query_count, self.candidate_topk)
        )
        if visual_inputs is None or visual_inputs.get('detail_images') is None:
            return (
                logits + self._zero_gradient_anchor(logits),
                selected_mask,
                empty_candidates,
                empty_residuals,
            )

        detail_images = visual_inputs['detail_images']
        detail_sizes = visual_inputs['detail_sizes']
        if detail_images.shape[0] != batch_size or detail_sizes.shape != (batch_size, 2):
            raise ValueError('CAR detail_images/detail_sizes do not match detector batch size.')

        batch_indices, query_indices = self._select_queries(logits)
        if batch_indices.numel() > 0:
            has_detail = (detail_sizes[batch_indices] > 0).all(dim=-1)
            batch_indices = batch_indices[has_detail]
            query_indices = query_indices[has_detail]
        if batch_indices.numel() == 0:
            return (
                logits + self._zero_gradient_anchor(logits),
                selected_mask,
                empty_candidates,
                empty_residuals,
            )

        rois = self._build_rois(boxes, batch_indices, query_indices, detail_sizes)
        crops = roi_align(
            detail_images,
            rois,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        detail_features = self.detail_encoder(crops)
        detail_tokens = detail_features.flatten(2).transpose(1, 2)

        candidate_indices = logits[batch_indices, query_indices].detach().topk(
            self.candidate_topk,
            dim=-1,
        ).indices
        prototypes = class_prototypes[candidate_indices]
        if self.use_difference_queries and self.candidate_topk > 1:
            other_mean = (
                prototypes.sum(dim=1, keepdim=True) - prototypes
            ) / (self.candidate_topk - 1)
            difference_queries = prototypes - other_mean
        else:
            difference_queries = prototypes

        attended, _ = self.detail_attn(
            self.prototype_norm(difference_queries),
            self.detail_norm(detail_tokens),
            detail_tokens,
            need_weights=False,
        )
        selected_queries = self.query_norm(query_features[batch_indices, query_indices])
        selected_queries = selected_queries.unsqueeze(1).expand(-1, self.candidate_topk, -1)
        candidate_residuals = self.residual_head(
            torch.cat([selected_queries, difference_queries, attended], dim=-1)
        ).squeeze(-1)

        flat_residual = logits.new_zeros(batch_size * query_count * class_count)
        flat_indices = (
            (batch_indices[:, None] * query_count + query_indices[:, None]) * class_count
            + candidate_indices
        ).reshape(-1)
        flat_residual = flat_residual.scatter_add(
            0,
            flat_indices,
            candidate_residuals.reshape(-1).to(dtype=logits.dtype),
        )
        residual = flat_residual.view(batch_size, query_count, class_count)

        selected_mask[batch_indices, query_indices] = True
        empty_candidates[batch_indices, query_indices] = candidate_indices
        empty_residuals[batch_indices, query_indices] = candidate_residuals
        return logits + residual, selected_mask, empty_candidates, empty_residuals


class MassConservingOptimalTransport(nn.Module):
    """Associate non-spatial mass events with detector queries under capacity constraints."""

    def __init__(
        self,
        num_classes,
        physics_stats_path,
        beta=1.0,
        epsilon=0.05,
        tau=1.0,
        sinkhorn_iterations=30,
        query_dustbin_cost=0.0,
        event_dustbin_cost=4.0,
        sensor_sigma_floor_g=0.25,
        evidence_clip=6.0,
        min_match_mass=0.05,
        reject_energy_threshold=-1.0,
        counterfactual_training=True,
        transport_mode='uot',
        use_event_dustbin=True,
        heteroscedastic=True,
        scene_prior_only=False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.epsilon = max(float(epsilon), 1e-6)
        self.tau = max(float(tau), 1e-6)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.query_dustbin_cost = float(query_dustbin_cost)
        self.event_dustbin_cost = float(event_dustbin_cost)
        self.sensor_sigma_floor_g = float(sensor_sigma_floor_g)
        self.evidence_clip = float(evidence_clip)
        self.min_match_mass = float(min_match_mass)
        self.reject_energy_threshold = float(reject_energy_threshold)
        self.counterfactual_training = bool(counterfactual_training)

        if transport_mode not in ('uot', 'balanced', 'greedy'):
            raise ValueError('transport_mode must be uot, balanced, or greedy.')
        self.transport_mode = transport_mode
        self.use_event_dustbin = bool(use_event_dustbin)
        self.heteroscedastic = bool(heteroscedastic)
        self.scene_prior_only = bool(scene_prior_only)

        stats = self._load_stats(physics_stats_path)
        feature_names = stats['feature_names']
        expected = [
            'delta_weight_g',
            'stable_before_g',
            'stable_after_g',
            'stable_std_g',
            'stable_range_g',
            'stable_drift_g',
            'stable_samples',
        ]
        if feature_names != expected:
            raise ValueError(f'MassOT requires feature order {expected}, got {feature_names}.')
        class_stats = sorted(
            stats['class_delta_gaussians'],
            key=lambda item: int(item['category_id']),
        )
        class_ids = [int(item['category_id']) for item in class_stats]
        if class_ids != list(range(self.num_classes)):
            raise ValueError('MassOT stats must contain contiguous category ids.')
        self.register_buffer(
            'class_mean_g',
            torch.tensor([abs(float(item['mean_g'])) for item in class_stats], dtype=torch.float32),
        )
        self.register_buffer(
            'class_std_g',
            torch.tensor([max(float(item['std_g']), 1e-3) for item in class_stats], dtype=torch.float32),
        )

    @staticmethod
    def _load_stats(path):
        stats_path = Path(path)
        if not stats_path.is_file():
            raise FileNotFoundError(f'MassOT statistics not found: {stats_path}')
        with stats_path.open('r', encoding='utf-8') as stream:
            stats = json.load(stream)
        required = {'feature_names', 'class_delta_gaussians'}
        missing = required.difference(stats)
        if missing:
            raise ValueError(f'MassOT statistics missing keys: {sorted(missing)}')
        return stats

    def _event_log_probabilities(self, raw_tokens, mask):
        raw_tokens = raw_tokens.float()
        delta = raw_tokens[..., 0].abs()
        before = raw_tokens[..., 1]
        after = raw_tokens[..., 2]
        stable_std = raw_tokens[..., 3].abs()
        stable_range = raw_tokens[..., 4].abs()
        stable_drift = raw_tokens[..., 5].abs()
        samples = raw_tokens[..., 6].abs().clamp_min(1.0)
        consistency = ((after - before).abs() - delta).abs()
        event_variance = torch.full_like(delta, self.sensor_sigma_floor_g ** 2)
        if self.heteroscedastic:
            event_variance = event_variance + (
                (stable_std / samples.sqrt()).square()
                + (stable_range / 4.0).square()
                + stable_drift.square()
                + (0.25 * consistency).square()
            )
        total_std = torch.sqrt(
            self.class_std_g.view(1, 1, -1).square()
            + event_variance.unsqueeze(-1)
        ).clamp_min(1e-4)
        z = (delta.unsqueeze(-1) - self.class_mean_g.view(1, 1, -1)) / total_std
        log_likelihood = -0.5 * z.square() - total_std.log() - 0.5 * math.log(2.0 * math.pi)
        log_probability = F.log_softmax(log_likelihood, dim=-1)
        return log_probability.masked_fill(~mask.unsqueeze(-1), -1e4)

    def _sinkhorn(self, cost):
        query_count, event_count = cost.shape
        event_dustbin_cost = (
            self.event_dustbin_cost if self.use_event_dustbin else 1e4
        )
        if self.transport_mode == 'greedy':
            plan = cost.new_zeros((query_count + 1, event_count + 1))
            plan[:query_count, event_count] = 1.0
            available = torch.ones(query_count, dtype=torch.bool, device=cost.device)
            event_order = cost.min(dim=0).values.argsort()
            for event_index in event_order.tolist():
                candidate_cost = cost[:, event_index].masked_fill(~available, torch.inf)
                best_cost, query_index = candidate_cost.min(dim=0)
                if torch.isfinite(best_cost) and (
                    not self.use_event_dustbin or best_cost < event_dustbin_cost
                ):
                    plan[query_index, event_index] = 1.0
                    plan[query_index, event_count] = 0.0
                    available[query_index] = False
                else:
                    plan[query_count, event_index] = 1.0
            plan[query_count, event_count] = float(available.sum())
            return plan

        augmented = cost.new_zeros((query_count + 1, event_count + 1))
        augmented[:query_count, :event_count] = cost
        augmented[:query_count, event_count] = self.query_dustbin_cost
        augmented[query_count, :event_count] = event_dustbin_cost

        row_mass = cost.new_ones(query_count + 1)
        column_mass = cost.new_ones(event_count + 1)
        row_mass[-1] = float(event_count)
        column_mass[-1] = float(query_count)
        log_row_mass = row_mass.clamp_min(1e-8).log()
        log_column_mass = column_mass.clamp_min(1e-8).log()
        log_kernel = -augmented / self.epsilon
        log_u = torch.zeros_like(log_row_mass)
        log_v = torch.zeros_like(log_column_mass)
        relaxation = 1.0 if self.transport_mode == 'balanced' else (
            self.tau / (self.tau + self.epsilon)
        )
        if not self.use_event_dustbin:
            # The no-dustbin ablation must account for every event. A relaxed
            # event marginal would silently destroy event mass instead.
            relaxation = 1.0
        for _ in range(self.sinkhorn_iterations):
            log_u = relaxation * (
                log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            )
            log_v = relaxation * (
                log_column_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
            )
        log_plan = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        return log_plan.clamp(min=-60.0, max=20.0).exp()

    @staticmethod
    def _cap_transport_plan(plan):
        """Project real transport onto per-query and per-event unit capacities."""
        for _ in range(3):
            row_mass = plan.sum(dim=1, keepdim=True)
            plan = plan / row_mass.clamp_min(1.0)
            column_mass = plan.sum(dim=0, keepdim=True)
            plan = plan / column_mass.clamp_min(1.0)
        return plan

    def _associate(self, visual_logits, event_log_probability, event_mask):
        batch_size, query_count, _ = visual_logits.shape
        max_events = event_mask.shape[1]
        plans = visual_logits.new_zeros((batch_size, query_count, max_events), dtype=torch.float32)
        query_dustbin = visual_logits.new_zeros((batch_size, query_count), dtype=torch.float32)
        event_dustbin = visual_logits.new_zeros((batch_size, max_events), dtype=torch.float32)
        energies = visual_logits.new_zeros((batch_size,), dtype=torch.float32)

        visual_probability = visual_logits.float().sigmoid().clamp_min(1e-8)
        visual_probability = visual_probability / visual_probability.sum(dim=-1, keepdim=True)
        event_probability = event_log_probability.exp()

        for batch_index in range(batch_size):
            valid_indices = event_mask[batch_index].nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            current_event_probability = event_probability[batch_index, valid_indices]
            compatibility = visual_probability[batch_index] @ current_event_probability.transpose(0, 1)
            cost = -compatibility.clamp_min(1e-8).log()
            augmented_plan = self._sinkhorn(cost)
            plan = augmented_plan[:query_count, :valid_indices.numel()]
            if not self.use_event_dustbin:
                plan = plan / plan.sum(dim=0, keepdim=True).clamp_min(1e-8)
            plan = self._cap_transport_plan(plan)
            current_query_dustbin = (1.0 - plan.sum(dim=1)).clamp(min=0.0, max=1.0)
            if self.use_event_dustbin:
                current_event_dustbin = (1.0 - plan.sum(dim=0)).clamp(min=0.0, max=1.0)
            else:
                current_event_dustbin = plan.new_zeros((valid_indices.numel(),))
            plans[batch_index, :, valid_indices] = plan
            query_dustbin[batch_index] = current_query_dustbin
            event_dustbin[batch_index, valid_indices] = current_event_dustbin

            transported = plan.sum()
            rejected = current_event_dustbin.sum()
            energy_numerator = (
                (plan * cost).sum() + rejected * (self.event_dustbin_cost if self.use_event_dustbin else 1e4)
            )
            energies[batch_index] = energy_numerator / (transported + rejected).clamp_min(1e-8)
        return plans, query_dustbin, event_dustbin, energies

    def forward(self, visual_logits, modal_inputs):
        output_dtype = visual_logits.dtype
        with torch.autocast(device_type=visual_logits.device.type, enabled=False):
            result = self._forward_float32(visual_logits.float(), modal_inputs)
        result['pred_logits'] = result['pred_logits'].to(dtype=output_dtype)
        return result

    def _forward_float32(self, visual_logits, modal_inputs):
        """Run physical likelihoods and transport in FP32 even under outer AMP."""
        batch_size, query_count, _ = visual_logits.shape
        if modal_inputs is None or modal_inputs.get('weight_tokens_raw') is None:
            return {
                'pred_logits': visual_logits,
                'transport_plan': visual_logits.new_zeros((batch_size, query_count, 0)),
                'query_dustbin': visual_logits.new_ones((batch_size, query_count)),
                'event_dustbin': visual_logits.new_zeros((batch_size, 0)),
                'transport_energy': visual_logits.new_zeros((batch_size,)),
                'match_mass': visual_logits.new_zeros((batch_size, query_count)),
                'event_mask': torch.zeros((batch_size, 0), dtype=torch.bool, device=visual_logits.device),
            }

        raw_tokens = modal_inputs['weight_tokens_raw'].to(device=visual_logits.device, dtype=torch.float32)
        event_mask = modal_inputs.get('weight_mask')
        if event_mask is None:
            event_mask = torch.ones(raw_tokens.shape[:2], dtype=torch.bool, device=visual_logits.device)
        else:
            event_mask = event_mask.to(device=visual_logits.device, dtype=torch.bool)
        if raw_tokens.ndim != 3 or raw_tokens.shape[0] != batch_size or raw_tokens.shape[-1] != 7:
            raise ValueError('weight_tokens_raw must have shape [batch, events, 7].')
        if event_mask.shape != raw_tokens.shape[:2]:
            raise ValueError('weight_mask must have shape [batch, events].')
        if raw_tokens.shape[1] == 0:
            return {
                'pred_logits': visual_logits,
                'transport_plan': visual_logits.new_zeros((batch_size, query_count, 0)),
                'query_dustbin': visual_logits.new_ones((batch_size, query_count)),
                'event_dustbin': visual_logits.new_zeros((batch_size, 0)),
                'transport_energy': visual_logits.new_zeros((batch_size,)),
                'match_mass': visual_logits.new_zeros((batch_size, query_count)),
                'event_mask': event_mask,
            }
        event_log_probability = self._event_log_probabilities(raw_tokens, event_mask)
        if self.scene_prior_only:
            event_probability = event_log_probability.exp().masked_fill(
                ~event_mask.unsqueeze(-1),
                0.0,
            )
            scene_probability = event_probability.max(dim=1).values
            has_weight = event_mask.any(dim=1)
            uniform = torch.full_like(scene_probability, 1.0 / self.num_classes)
            scene_probability = torch.where(
                has_weight.unsqueeze(-1),
                scene_probability,
                uniform,
            )
            evidence = (scene_probability.clamp_min(1e-8).log() + math.log(self.num_classes))
            evidence = evidence.clamp(min=-self.evidence_clip, max=self.evidence_clip)
            return {
                'pred_logits': visual_logits + self.beta * evidence.to(visual_logits.dtype).unsqueeze(1),
                'transport_plan': visual_logits.new_zeros((batch_size, query_count, raw_tokens.shape[1])),
                'query_dustbin': visual_logits.new_ones((batch_size, query_count)),
                'event_dustbin': visual_logits.new_zeros(raw_tokens.shape[:2]),
                'transport_energy': visual_logits.new_zeros((batch_size,)),
                'match_mass': visual_logits.new_zeros((batch_size, query_count)),
                'event_mask': event_mask,
                'weight_event_log_probability': event_log_probability,
            }
        plan, query_dustbin, event_dustbin, energy = self._associate(
            visual_logits,
            event_log_probability,
            event_mask,
        )

        match_mass = plan.sum(dim=-1)
        weighted_log_probability = torch.matmul(plan, event_log_probability.masked_fill(
            ~event_mask.unsqueeze(-1),
            0.0,
        ))
        weighted_log_probability = weighted_log_probability / match_mass.unsqueeze(-1).clamp_min(1e-8)
        evidence = weighted_log_probability + math.log(self.num_classes)
        evidence = evidence.clamp(min=-self.evidence_clip, max=self.evidence_clip)
        accepted = match_mass >= self.min_match_mass
        if self.reject_energy_threshold >= 0:
            accepted = accepted & (energy <= self.reject_energy_threshold).unsqueeze(-1)
        accepted = accepted & event_mask.any(dim=1, keepdim=True)
        update_strength = match_mass.clamp(0.0, 1.0) * accepted.to(dtype=match_mass.dtype)
        refined_logits = visual_logits.float() + self.beta * update_strength.unsqueeze(-1) * evidence

        result = {
            'pred_logits': refined_logits.to(dtype=visual_logits.dtype),
            'transport_plan': plan,
            'query_dustbin': query_dustbin,
            'event_dustbin': event_dustbin,
            'transport_energy': energy,
            'match_mass': match_mass,
            'event_mask': event_mask,
            'weight_event_log_probability': event_log_probability,
        }
        if self.training and self.counterfactual_training and batch_size > 1:
            negative_log_probability = event_log_probability.roll(shifts=1, dims=0)
            negative_mask = event_mask.roll(shifts=1, dims=0)
            _, _, _, negative_energy = self._associate(
                visual_logits,
                negative_log_probability,
                negative_mask,
            )
            result['negative_transport_energy'] = negative_energy
        return result


@register()
class CARMassOTRTDETR(RTDETR):
    """RT-DETRv2 wrapper with optional CAR and MassOT classification refinement."""

    __inject__ = ['backbone', 'encoder', 'decoder']
    __share__ = ['num_classes']

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes=80,
        enable_car=True,
        enable_massot=False,
        physics_stats_path=None,
        freeze_detector=False,
        car_train_max_queries=30,
        car_eval_max_queries=10,
        car_candidate_topk=3,
        car_use_difference_queries=True,
        car_crop_size=160,
        car_box_expand_ratio=0.15,
        car_min_objectness=0.05,
        car_max_top2_margin=0.35,
        massot_beta=1.0,
        massot_epsilon=0.05,
        massot_tau=1.0,
        massot_sinkhorn_iterations=30,
        massot_query_dustbin_cost=0.0,
        massot_event_dustbin_cost=4.0,
        massot_sensor_sigma_floor_g=0.25,
        massot_evidence_clip=6.0,
        massot_min_match_mass=0.05,
        massot_reject_energy_threshold=-1.0,
        massot_counterfactual_training=True,
        massot_transport_mode='uot',
        massot_use_event_dustbin=True,
        massot_heteroscedastic=True,
        massot_scene_prior_only=False,
    ):
        super().__init__(backbone=backbone, encoder=encoder, decoder=decoder)
        self.enable_car = bool(enable_car)
        self.enable_massot = bool(enable_massot)
        self.freeze_detector = bool(freeze_detector)
        hidden_dim = int(getattr(decoder, 'hidden_dim', 256))

        if self.freeze_detector:
            self.backbone.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)

        if self.enable_car:
            self.car = ConfusionAwareReinspection(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                nhead=int(getattr(decoder, 'nhead', 8)),
                train_max_queries=car_train_max_queries,
                eval_max_queries=car_eval_max_queries,
                candidate_topk=car_candidate_topk,
                use_difference_queries=car_use_difference_queries,
                crop_size=car_crop_size,
                box_expand_ratio=car_box_expand_ratio,
                min_objectness=car_min_objectness,
                max_top2_margin=car_max_top2_margin,
            )
        else:
            self.car = None

        if self.enable_massot:
            if not physics_stats_path:
                raise ValueError('physics_stats_path is required when enable_massot=True.')
            self.massot = MassConservingOptimalTransport(
                num_classes=num_classes,
                physics_stats_path=physics_stats_path,
                beta=massot_beta,
                epsilon=massot_epsilon,
                tau=massot_tau,
                sinkhorn_iterations=massot_sinkhorn_iterations,
                query_dustbin_cost=massot_query_dustbin_cost,
                event_dustbin_cost=massot_event_dustbin_cost,
                sensor_sigma_floor_g=massot_sensor_sigma_floor_g,
                evidence_clip=massot_evidence_clip,
                min_match_mass=massot_min_match_mass,
                reject_energy_threshold=massot_reject_energy_threshold,
                counterfactual_training=massot_counterfactual_training,
                transport_mode=massot_transport_mode,
                use_event_dustbin=massot_use_event_dustbin,
                heteroscedastic=massot_heteroscedastic,
                scene_prior_only=massot_scene_prior_only,
            )
        else:
            self.massot = None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_detector:
            self.backbone.eval()
            self.encoder.eval()
            self.decoder.eval()
        return self

    def forward(self, x, targets=None, modal_inputs=None, visual_inputs=None):
        features = self.backbone(x)
        features = self.encoder(features)
        outputs = self.decoder(features, targets=targets)
        visual_logits = outputs['pred_logits']
        outputs['pred_logits_visual'] = visual_logits

        car_logits = visual_logits
        if self.car is not None:
            if 'pred_query_features' not in outputs or 'class_prototypes' not in outputs:
                raise KeyError('CAR requires decoder expose_query_features=True.')
            car_logits, selected, candidates, residuals = self.car(
                visual_logits,
                outputs['pred_boxes'],
                outputs['pred_query_features'],
                outputs['class_prototypes'],
                visual_inputs,
            )
            outputs['car_query_mask'] = selected
            outputs['car_candidate_indices'] = candidates
            outputs['car_candidate_residuals'] = residuals
        outputs['pred_logits_car'] = car_logits

        final_logits = car_logits
        if self.massot is not None:
            massot_outputs = self.massot(car_logits, modal_inputs)
            final_logits = massot_outputs.pop('pred_logits')
            outputs['transport_plan'] = massot_outputs.pop('transport_plan')
            outputs['transport_energy'] = massot_outputs.pop('transport_energy')
            outputs['massot_match_mass'] = massot_outputs.pop('match_mass')
            outputs['massot_query_dustbin'] = massot_outputs.pop('query_dustbin')
            outputs['massot_event_dustbin'] = massot_outputs.pop('event_dustbin')
            outputs['massot_event_mask'] = massot_outputs.pop('event_mask')
            outputs.update({f'massot_{key}': value for key, value in massot_outputs.items()})
        outputs['pred_logits'] = final_logits
        return outputs