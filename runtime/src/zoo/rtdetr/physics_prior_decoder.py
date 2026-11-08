"""Physics-grounded scene priors for multimodal RT-DETRv2."""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoising import get_contrastive_denoising_training_group
from .rtdetrv2_decoder import RTDETRTransformerv2
from ...core import register


__all__ = ['PhysicsReliabilityWeightPriorMMRTDETRTransformerv2']


@register()
class PhysicsReliabilityWeightPriorMMRTDETRTransformerv2(RTDETRTransformerv2):
    """Add a reliability-aware physical weight prior to classification logits.

    The fixed Gaussian likelihood is estimated from the training split only.
    Its scene-level evidence is shared by all object queries, while the box
    branch remains purely visual. Optional counterfactual training teaches the
    reliability gate to reject batch-shuffled weight observations.
    """

    def __init__(
        self,
        *args,
        physics_stats_path,
        weight_input_dim=7,
        physics_prior_strength=2.0,
        physics_evidence_clip=6.0,
        use_quality_gate=True,
        use_agreement_gate=False,
        detach_visual_agreement=True,
        reliability_hidden_dim=64,
        reliability_init_probability=0.95,
        quality_feature_indices=None,
        modality_dropout_probability=0.0,
        counterfactual_gate_weight=0.0,
        counterfactual_prior_weight=0.0,
        counterfactual_min_js=0.05,
        use_residual_correction=False,
        residual_hidden_dim=64,
        residual_max_logit=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_input_dim = int(weight_input_dim)
        self.physics_prior_strength = float(physics_prior_strength)
        self.physics_evidence_clip = float(physics_evidence_clip)
        self.use_quality_gate = bool(use_quality_gate)
        self.use_agreement_gate = bool(use_agreement_gate)
        self.detach_visual_agreement = bool(detach_visual_agreement)
        self.modality_dropout_probability = float(modality_dropout_probability)
        self.counterfactual_gate_weight = float(counterfactual_gate_weight)
        self.counterfactual_prior_weight = float(counterfactual_prior_weight)
        self.counterfactual_min_js = float(counterfactual_min_js)
        self.use_residual_correction = bool(use_residual_correction)
        self.residual_max_logit = float(residual_max_logit)

        if quality_feature_indices is None:
            quality_feature_indices = [3, 4, 5, 6]
        self.quality_feature_indices = tuple(int(i) for i in quality_feature_indices)
        if any(i < 0 or i >= self.weight_input_dim for i in self.quality_feature_indices):
            raise ValueError('quality_feature_indices must index weight token features')
        if not 0.0 <= self.modality_dropout_probability < 1.0:
            raise ValueError('modality_dropout_probability must be in [0, 1)')

        stats = self._load_physics_stats(physics_stats_path)
        feature_names = stats['feature_names']
        if len(feature_names) != self.weight_input_dim:
            raise ValueError(
                f'weight_input_dim={self.weight_input_dim} does not match '
                f'{len(feature_names)} features in {physics_stats_path}'
            )
        if feature_names[0] != 'delta_weight_g':
            raise ValueError('delta_weight_g must be the first weight feature')

        class_stats = sorted(stats['class_delta_gaussians'], key=lambda item: int(item['category_id']))
        class_ids = [int(item['category_id']) for item in class_stats]
        if class_ids != list(range(self.num_classes)):
            raise ValueError('physics stats must contain contiguous category ids [0, num_classes)')

        delta_global_mean = float(stats['feature_stats']['delta_weight_g']['mean'])
        delta_global_std = max(float(stats['feature_stats']['delta_weight_g']['std']), 1e-6)
        class_mean = [
            (float(item['mean_g']) - delta_global_mean) / delta_global_std
            for item in class_stats
        ]
        class_std = [
            max(float(item['std_g']) / delta_global_std, 1e-6)
            for item in class_stats
        ]
        self.register_buffer('physics_class_mean', torch.tensor(class_mean, dtype=torch.float32))
        self.register_buffer('physics_class_std', torch.tensor(class_std, dtype=torch.float32))
        self.register_buffer(
            'uniform_class_probability',
            torch.tensor(1.0 / self.num_classes, dtype=torch.float32),
        )

        quality_dim = 2 * len(self.quality_feature_indices) + 3
        agreement_dim = 4 if self.use_agreement_gate else 0
        gate_dim = quality_dim + agreement_dim
        if self.use_quality_gate:
            hidden = max(8, int(reliability_hidden_dim))
            self.reliability_gate = nn.Sequential(
                nn.LayerNorm(gate_dim),
                nn.Linear(gate_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            init_probability = min(max(float(reliability_init_probability), 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init_probability / (1.0 - init_probability))
            nn.init.zeros_(self.reliability_gate[-1].weight)
            nn.init.constant_(self.reliability_gate[-1].bias, init_logit)
        else:
            self.reliability_gate = None

        if self.use_residual_correction:
            hidden = max(8, int(residual_hidden_dim))
            self.residual_event_head = nn.Sequential(
                nn.LayerNorm(self.weight_input_dim),
                nn.Linear(self.weight_input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_classes),
            )
            nn.init.zeros_(self.residual_event_head[-1].weight)
            nn.init.zeros_(self.residual_event_head[-1].bias)
        else:
            self.residual_event_head = None

    @staticmethod
    def _load_physics_stats(path):
        stats_path = Path(path)
        if not stats_path.is_file():
            raise FileNotFoundError(f'Physical prior stats not found: {stats_path}')
        with stats_path.open('r', encoding='utf-8') as stream:
            stats = json.load(stream)
        required = {'feature_names', 'feature_stats', 'class_delta_gaussians'}
        missing = required.difference(stats)
        if missing:
            raise ValueError(f'Physical prior stats missing keys: {sorted(missing)}')
        return stats

    def _prepare_modal_inputs(self, modal_inputs, batch_size, dtype, device):
        if modal_inputs is None or modal_inputs.get('weight_tokens') is None:
            tokens = torch.zeros(batch_size, 1, self.weight_input_dim, dtype=dtype, device=device)
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            return tokens, mask

        tokens = modal_inputs['weight_tokens'].to(device=device, dtype=dtype)
        if tokens.ndim != 3 or tokens.shape[0] != batch_size or tokens.shape[-1] != self.weight_input_dim:
            raise ValueError(
                f'weight_tokens must have shape [B, M, {self.weight_input_dim}], got {tuple(tokens.shape)}'
            )
        mask = modal_inputs.get('weight_mask')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError('weight_mask must have shape [B, M]')
        return tokens, mask

    def _event_physics_logits(self, tokens, mask):
        delta = tokens[..., 0].unsqueeze(-1)
        mean = self.physics_class_mean.to(dtype=tokens.dtype).view(1, 1, -1)
        std = self.physics_class_std.to(dtype=tokens.dtype).view(1, 1, -1)
        logits = -0.5 * ((delta - mean) / std).square() - torch.log(std)
        if self.residual_event_head is not None:
            residual = torch.tanh(self.residual_event_head(tokens)) * self.residual_max_logit
            logits = logits + residual
        return logits.masked_fill(~mask.unsqueeze(-1), -1e4)

    def _scene_physics_distribution(self, event_logits, mask):
        event_probability = F.softmax(event_logits, dim=-1)
        event_probability = event_probability.masked_fill(~mask.unsqueeze(-1), 0.0)
        scene_probability = event_probability.max(dim=1).values
        has_weight = mask.any(dim=1)
        uniform = self.uniform_class_probability.to(dtype=scene_probability.dtype)
        scene_probability = torch.where(
            has_weight.unsqueeze(-1),
            scene_probability,
            torch.full_like(scene_probability, uniform),
        )
        scene_probability = scene_probability.clamp_min(1e-8)
        scene_distribution = scene_probability / scene_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = torch.log(scene_probability / uniform)
        if self.physics_evidence_clip > 0:
            evidence = evidence.clamp(-self.physics_evidence_clip, self.physics_evidence_clip)
        return event_probability, scene_distribution, evidence, has_weight

    @staticmethod
    def _entropy(distribution):
        distribution = distribution.clamp_min(1e-8)
        return -(distribution * distribution.log()).sum(dim=-1, keepdim=True)

    @staticmethod
    def _js_divergence(first, second):
        first = first.clamp_min(1e-8)
        second = second.clamp_min(1e-8)
        middle = 0.5 * (first + second)
        return 0.5 * (
            (first * (first.log() - middle.log())).sum(dim=-1, keepdim=True)
            + (second * (second.log() - middle.log())).sum(dim=-1, keepdim=True)
        )

    def _visual_distribution(self, visual_logits):
        visual_probability = torch.sigmoid(visual_logits).max(dim=1).values
        visual_distribution = visual_probability / visual_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.detach_visual_agreement:
            visual_distribution = visual_distribution.detach()
        return visual_distribution

    def _quality_features(self, tokens, mask):
        valid = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        selected = tokens[..., self.quality_feature_indices]
        selected_abs = selected.abs()
        quality_mean = (selected_abs * valid).sum(dim=1) / count
        quality_max = selected_abs.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        quality_max = torch.where(mask.any(dim=1, keepdim=True), quality_max, torch.zeros_like(quality_max))

        delta = tokens[..., 0].unsqueeze(-1)
        mean = self.physics_class_mean.to(dtype=tokens.dtype).view(1, 1, -1)
        std = self.physics_class_std.to(dtype=tokens.dtype).view(1, 1, -1)
        nearest_z = ((delta - mean).abs() / std).min(dim=-1).values
        nearest_z_mean = (nearest_z * mask.to(dtype=tokens.dtype)).sum(dim=1, keepdim=True) / count
        nearest_z_max = nearest_z.masked_fill(~mask, -1e4).max(dim=1, keepdim=True).values
        nearest_z_max = torch.where(mask.any(dim=1, keepdim=True), nearest_z_max, torch.zeros_like(nearest_z_max))
        event_count = mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
        event_count = torch.log1p(event_count)
        return torch.cat(
            [quality_mean, quality_max, nearest_z_mean, nearest_z_max, event_count],
            dim=-1,
        )

    def _build_reliability(self, tokens, mask, scene_distribution, visual_distribution):
        has_weight = mask.any(dim=1, keepdim=True)
        if self.reliability_gate is None:
            return has_weight.to(dtype=tokens.dtype), None

        features = [self._quality_features(tokens, mask)]
        if self.use_agreement_gate:
            js = self._js_divergence(scene_distribution, visual_distribution)
            overlap = torch.minimum(scene_distribution, visual_distribution).sum(dim=-1, keepdim=True)
            visual_entropy = self._entropy(visual_distribution) / math.log(self.num_classes)
            weight_entropy = self._entropy(scene_distribution) / math.log(self.num_classes)
            features.extend([js, overlap, visual_entropy, weight_entropy])
        gate_features = torch.cat(features, dim=-1)
        reliability = torch.sigmoid(self.reliability_gate(gate_features))
        return reliability * has_weight.to(dtype=reliability.dtype), gate_features

    def _counterfactual_losses(
        self,
        tokens,
        mask,
        scene_distribution,
        visual_distribution,
        reliability,
    ):
        if tokens.shape[0] < 2 or self.reliability_gate is None:
            return {}
        if self.counterfactual_gate_weight <= 0 and self.counterfactual_prior_weight <= 0:
            return {}

        mismatch_tokens = tokens.roll(shifts=1, dims=0)
        mismatch_mask = mask.roll(shifts=1, dims=0)
        mismatch_event_logits = self._event_physics_logits(mismatch_tokens, mismatch_mask)
        _, mismatch_distribution, mismatch_evidence, mismatch_has_weight = \
            self._scene_physics_distribution(mismatch_event_logits, mismatch_mask)
        mismatch_reliability, _ = self._build_reliability(
            mismatch_tokens,
            mismatch_mask,
            mismatch_distribution,
            visual_distribution,
        )

        counterfactual_js = self._js_divergence(scene_distribution.detach(), mismatch_distribution.detach())
        valid = mismatch_has_weight.unsqueeze(-1) & (counterfactual_js >= self.counterfactual_min_js)
        valid_float = valid.to(dtype=reliability.dtype)
        denominator = valid_float.sum().clamp_min(1.0)
        losses = {}
        if self.counterfactual_gate_weight > 0:
            paired_logit = torch.logit(reliability.float().clamp(1e-6, 1.0 - 1e-6))
            mismatch_logit = torch.logit(
                mismatch_reliability.float().clamp(1e-6, 1.0 - 1e-6)
            )
            positive_loss = F.binary_cross_entropy_with_logits(
                paired_logit,
                torch.ones_like(paired_logit),
                reduction='none',
            )
            negative_loss = F.binary_cross_entropy_with_logits(
                mismatch_logit,
                torch.zeros_like(mismatch_logit),
                reduction='none',
            )
            paired = (positive_loss + negative_loss) * valid_float
            losses['loss_counterfactual_gate'] = \
                self.counterfactual_gate_weight * paired.sum() / denominator
        if self.counterfactual_prior_weight > 0:
            leakage = mismatch_reliability * mismatch_evidence.abs().mean(dim=-1, keepdim=True)
            losses['loss_counterfactual_prior'] = \
                self.counterfactual_prior_weight * (leakage * valid_float).sum() / denominator
        return losses

    def forward(self, feats, targets=None, modal_inputs=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        tokens, weight_mask = self._prepare_modal_inputs(
            modal_inputs,
            batch_size=out_logits.shape[1],
            dtype=out_logits.dtype,
            device=out_logits.device,
        )
        event_logits = self._event_physics_logits(tokens, weight_mask)
        _, scene_distribution, evidence, has_weight = \
            self._scene_physics_distribution(event_logits, weight_mask)
        visual_distribution = self._visual_distribution(out_logits[-1])
        reliability, gate_features = self._build_reliability(
            tokens,
            weight_mask,
            scene_distribution,
            visual_distribution,
        )

        training_aux_losses = {}
        if self.training:
            training_aux_losses.update(
                self._counterfactual_losses(
                    tokens,
                    weight_mask,
                    scene_distribution,
                    visual_distribution,
                    reliability,
                )
            )
            if self.modality_dropout_probability > 0:
                keep = torch.rand_like(reliability) >= self.modality_dropout_probability
                reliability = reliability * keep.to(dtype=reliability.dtype)

        prior_update = self.physics_prior_strength * reliability * evidence
        out_logits = out_logits + prior_update.unsqueeze(0).unsqueeze(2)

        out = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1],
            'weight_event_logits': event_logits,
            'weight_scene_distribution': scene_distribution,
            'weight_reliability': reliability,
            'weight_has_observation': has_weight,
        }
        if gate_features is not None:
            out['weight_gate_features'] = gate_features
        if training_aux_losses:
            out['training_aux_losses'] = training_aux_losses

        if self.training and self.expose_query_features:
            out['pred_query_features'] = out_features[-1]
            out['class_prototypes'] = self.dec_score_head[-1].weight

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out
