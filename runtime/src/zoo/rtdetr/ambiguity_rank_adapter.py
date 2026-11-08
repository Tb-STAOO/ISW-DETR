"""A shared, bounded adapter for observable visual/mass ambiguity.

The same MLP is applied to every query/class pair.  Its inputs contain neither
class IDs, labels, event order, nor ground-truth record ownership.  The final
layer starts at zero, preserving the base detector before adapter training.
"""

import math

import torch
from torch import nn


class _ExactIdentityResidualAdd(torch.autograd.Function):
    """Add a residual while preserving exact input bits when the residual is zero.

    At zero, a normal floating-point addition can turn -0 into +0.  Selecting
    the original value preserves it.  The mathematical operation remains an
    addition, so both inputs receive its ordinary gradient, including at zero.
    """

    @staticmethod
    def forward(ctx, base, delta):
        return torch.where(delta == 0, base, base + delta)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class MassGuidedDenoisingRankAdapter(nn.Module):
    FEATURE_VERSION = "observable_visual_mass_v1"
    FEATURE_NAMES = (
        "visual_logit_div10", "base_logit_div10",
        "visual_relative_max_div10", "base_relative_max_div10",
        "visual_class_softmax", "base_class_softmax",
        "scene_evidence_div6", "record_residual_div6",
        "visual_max_sigmoid", "base_max_sigmoid",
    )

    def __init__(self, bound=2.0, enabled=True):
        super().__init__()
        self.bound = float(bound)
        self.enabled = bool(enabled)
        if not math.isfinite(self.bound) or self.bound <= 0:
            raise ValueError("bound must be finite and positive.")
        self.mlp = nn.Sequential(
            nn.Linear(10, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _validate(visual, base, scene, residual, candidates, gate):
        if base.ndim != 3 or not base.is_floating_point() or base.shape[-1] == 0:
            raise ValueError("base_logits must be floating point [B,Q,C], with C > 0.")
        for name, value in (("visual_logits", visual), ("record_residual", residual)):
            if value.shape != base.shape or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point with shape [B,Q,C].")
        if scene.shape != (base.shape[0], base.shape[2]) or not scene.is_floating_point():
            raise ValueError("scene_evidence must be floating point [B,C].")
        if candidates.shape != base.shape[:2] or candidates.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean [B,Q].")
        if gate.shape != base.shape[:1] or gate.dtype != torch.bool:
            raise ValueError("integrity_gate must be boolean [B].")
        for name, value in (("visual_logits", visual), ("scene_evidence", scene),
                            ("record_residual", residual), ("candidate_mask", candidates),
                            ("integrity_gate", gate)):
            if value.device != base.device:
                raise ValueError(f"{name} must be on the same device as base_logits.")

    def forward(self, visual_logits, base_logits, scene_evidence, record_residual,
                candidate_mask, integrity_gate):
        self._validate(visual_logits, base_logits, scene_evidence, record_residual,
                       candidate_mask, integrity_gate)
        active = candidate_mask & integrity_gate[:, None]
        zero_delta = torch.zeros_like(base_logits)
        if not self.enabled or not bool(active.any().item()):
            return {
                "pred_logits": base_logits,
                "rank_adapter_delta": zero_delta,
                "rank_adapter_features": base_logits.new_zeros((*base_logits.shape, 10)),
                "rank_adapter_applied": torch.zeros_like(integrity_gate),
            }
        if self.mlp[0].weight.device != base_logits.device:
            raise ValueError("Move the adapter to the same device as base_logits before forward.")

        # Non-finite padding/rejected queries are ignored.  A non-finite value
        # in any actively used observation is a caller error, rather than a
        # silently fabricated feature or correction.
        active_class = active.unsqueeze(-1)
        observation_finite = (torch.isfinite(visual_logits) & torch.isfinite(base_logits)
                              & torch.isfinite(record_residual)
                              & torch.isfinite(scene_evidence).unsqueeze(1))
        if not bool((~active_class | observation_finite).all().item()):
            raise ValueError("Active rank-adapter observations must be finite.")

        dtype = self.mlp[0].weight.dtype
        visual = torch.where(active_class, visual_logits,
                             torch.zeros_like(visual_logits)).to(dtype=dtype)
        base = torch.where(active_class, base_logits,
                           torch.zeros_like(base_logits)).to(dtype=dtype)
        residual = torch.where(active_class, record_residual,
                               torch.zeros_like(record_residual)).to(dtype=dtype)
        scene = torch.where(integrity_gate[:, None], scene_evidence,
                            torch.zeros_like(scene_evidence)).to(dtype=dtype)
        features = torch.stack((
            visual / 10.0,
            base / 10.0,
            (visual - visual.amax(-1, keepdim=True)) / 10.0,
            (base - base.amax(-1, keepdim=True)) / 10.0,
            visual.softmax(-1),
            base.softmax(-1),
            (scene / 6.0).unsqueeze(1).expand_as(base),
            residual / 6.0,
            visual.sigmoid().amax(-1, keepdim=True).expand_as(base),
            base.sigmoid().amax(-1, keepdim=True).expand_as(base),
        ), dim=-1)
        features = torch.where(active_class.unsqueeze(-1), features,
                               torch.zeros_like(features))
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("Rank-adapter feature conversion produced a non-finite value.")
        score = self.mlp(features).squeeze(-1)
        if not bool(torch.isfinite(score[active]).all().item()):
            raise ValueError("Rank-adapter MLP produced a non-finite active score.")
        delta = self.bound * score.tanh()
        delta = torch.where(active_class, delta, torch.zeros_like(delta)).to(dtype=base_logits.dtype)
        output = _ExactIdentityResidualAdd.apply(base_logits, delta)
        return {
            "pred_logits": output,
            "rank_adapter_delta": delta,
            "rank_adapter_features": features,
            "rank_adapter_applied": active.any(-1),
        }


# Compatibility for early local callers; the canonical name makes no causal
# counterfactual claim about the training-time logit perturbation curriculum.
CounterfactualMassRankAdapter = MassGuidedDenoisingRankAdapter
