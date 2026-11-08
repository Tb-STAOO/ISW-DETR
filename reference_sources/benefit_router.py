"""Query-wise routing of a frozen CVSR proposal; no labels enter this module.

The scalar gate is in [0, 2]. A gate of one reproduces the proposal exactly,
while a gate of zero reproduces the visual input. Candidate selection and
record assignment are external, frozen operations. Padding is masked before
feature reduction, so absent or invalid optional telemetry cannot inject NaNs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


FEATURE_NAMES = (
    "visual_max_score", "visual_top2_gap", "visual_entropy",
    "visual_max_probability", "proposal_max_score", "proposal_top2_gap",
    "proposal_entropy", "top1_disagreement", "visual_probability_of_proposal",
    "proposal_probability_of_visual", "proposal_mean_absolute_change",
    "proposal_max_absolute_change", "scene_winner_support_difference",
    "record_residual_mean_absolute", "assigned_row_margin", "assigned_col_margin",
    "sensor_uncertainty", "record_class_entropy", "visual_top2_mass_separation",
    "proposal_visual_mass_difference",
)
VISUAL_FEATURE_INDICES = (0, 1, 2, 3)
MASS_FEATURE_INDICES = tuple(range(4, len(FEATURE_NAMES)))


def _require_shape(value: Tensor, shape, name: str) -> None:
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}.")


def _safe(value: Tensor) -> Tensor:
    return torch.nan_to_num(value.float(), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def _distribution(logits: Tensor):
    bounded = _safe(logits).clamp(-60, 60)
    probability = bounded.softmax(-1)
    top = bounded.topk(min(2, bounded.shape[-1]), dim=-1)
    winner = top.indices[..., 0]
    gap = top.values[..., 0] - top.values[..., -1]
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / max(math.log(bounded.shape[-1]), 1e-8)
    return bounded, probability, winner, gap, entropy, top.indices[..., -1]


def extract_router_features(
    visual_logits: Tensor,
    base_logits: Tensor,
    candidate_mask: Tensor,
    transaction_gate: Tensor,
    *,
    scene_evidence: Optional[Tensor] = None,
    event_log_probability: Optional[Tensor] = None,
    weight_tokens_raw: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    query_record_residual: Optional[Tensor] = None,
    hard_p: Optional[Tensor] = None,
    row_margin: Optional[Tensor] = None,
    col_margin: Optional[Tensor] = None,
    feature_mode: str = "full",
) -> Tensor:
    """Return [batch, queries, 20] predicted-only features.

    Optional shapes: scene [B,C], logp [B,E,C], raw [B,E,7], event
    mask [B,E], residual [B,Q,C], hard assignment [B,Q,E], margins [B,Q].
    ``no_mass_input`` retains only features of the original visual logits,
    including removal of every feature derived from the mass proposal.
    """
    if visual_logits.ndim != 3 or visual_logits.shape[-1] == 0:
        raise ValueError("visual_logits must have shape [B,Q,C] with at least one class.")
    batch, queries, classes = visual_logits.shape
    _require_shape(base_logits, visual_logits.shape, "base_logits")
    _require_shape(candidate_mask, (batch, queries), "candidate_mask")
    _require_shape(transaction_gate, (batch,), "transaction_gate")
    if feature_mode not in {"full", "no_mass_input"}:
        raise ValueError("feature_mode must be 'full' or 'no_mass_input'.")
    active = candidate_mask.bool() & transaction_gate.bool()[:, None]
    active = active & torch.isfinite(visual_logits).all(-1) & torch.isfinite(base_logits).all(-1)
    z, pz, winner_z, gap_z, entropy_z, second_z = _distribution(visual_logits)
    proposal, pb, winner_b, gap_b, entropy_b, _ = _distribution(base_logits)
    zero = z.new_zeros((batch, queries))
    delta = _safe(base_logits) - _safe(visual_logits)
    take = lambda values, index: values.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    values = [
        z.sigmoid().max(-1).values, (gap_z / 4).tanh(), entropy_z, pz.max(-1).values,
        proposal.sigmoid().max(-1).values, (gap_b / 4).tanh(), entropy_b,
        (winner_z != winner_b).float(), take(pz, winner_b), take(pb, winner_z),
        (delta.abs().mean(-1) / 4).tanh(), (delta.abs().max(-1).values / 4).tanh(),
    ]
    if scene_evidence is None:
        values.append(zero)
    else:
        _require_shape(scene_evidence, (batch, classes), "scene_evidence")
        scene = _safe(scene_evidence)[:, None].expand(-1, queries, -1)
        values.append(((take(scene, winner_b) - take(scene, winner_z)) / 4).tanh())
    if query_record_residual is None:
        values.append(zero)
    else:
        _require_shape(query_record_residual, (batch, queries, classes), "query_record_residual")
        values.append((_safe(query_record_residual).abs().mean(-1) / 4).tanh())
    for margin, name in ((row_margin, "row_margin"), (col_margin, "col_margin")):
        if margin is None:
            values.append(zero)
        else:
            _require_shape(margin, (batch, queries), name)
            values.append((_safe(margin) / 4).tanh())

    events = None
    if event_log_probability is not None:
        if event_log_probability.ndim != 3:
            raise ValueError("event_log_probability must have shape [B,E,C].")
        events = event_log_probability.shape[1]
        _require_shape(event_log_probability, (batch, events, classes), "event_log_probability")
    if weight_tokens_raw is not None:
        if weight_tokens_raw.ndim != 3:
            raise ValueError("weight_tokens_raw must have shape [B,E,7].")
        if events is None:
            events = weight_tokens_raw.shape[1]
        _require_shape(weight_tokens_raw, (batch, events, 7), "weight_tokens_raw")
    if weight_mask is not None:
        if weight_mask.ndim != 2:
            raise ValueError("weight_mask must have shape [B,E].")
        if events is None:
            events = weight_mask.shape[1]
        _require_shape(weight_mask, (batch, events), "weight_mask")
    if hard_p is not None:
        if hard_p.ndim != 3:
            raise ValueError("hard_p must have shape [B,Q,E].")
        if events is None:
            events = hard_p.shape[-1]
        _require_shape(hard_p, (batch, queries, events), "hard_p")
    events = 0 if events is None else events
    event_valid = torch.ones((batch, events), dtype=torch.bool, device=z.device)
    if weight_mask is not None:
        event_valid &= weight_mask.bool()
    if event_log_probability is not None:
        event_valid &= torch.isfinite(event_log_probability).all(-1)
    if weight_tokens_raw is not None:
        raw_valid = torch.isfinite(weight_tokens_raw).all(-1)
        raw_valid &= (weight_tokens_raw[..., 3:6] >= 0).all(-1) & (weight_tokens_raw[..., 6] >= 1)
        sensor_valid = event_valid & raw_valid
        raw = _safe(weight_tokens_raw)
        sigma = torch.sqrt(
            (raw[..., 3] / raw[..., 6].clamp_min(1).sqrt()).square()
            + (raw[..., 4] / 4).square() + raw[..., 5].square()
        )
        sensor = (sigma * sensor_valid).sum(-1) / sensor_valid.sum(-1).clamp_min(1)
        values.append((sensor / 2).tanh()[:, None].expand(-1, queries))
    else:
        values.append(zero)

    if event_log_probability is None or events == 0:
        values.extend((zero, zero, zero))
    else:
        lp = _safe(event_log_probability).clamp(-1e4, 1e4)
        ep = lp.softmax(-1)
        event_entropy = -(ep * ep.clamp_min(1e-12).log()).sum(-1)
        event_entropy = event_entropy / max(math.log(classes), 1e-8)
        lp_z = lp[:, None].expand(-1, queries, -1, -1).gather(
            -1, winner_z[:, :, None, None].expand(-1, -1, events, 1)).squeeze(-1)
        lp_second = lp[:, None].expand(-1, queries, -1, -1).gather(
            -1, second_z[:, :, None, None].expand(-1, -1, events, 1)).squeeze(-1)
        lp_b = lp[:, None].expand(-1, queries, -1, -1).gather(
            -1, winner_b[:, :, None, None].expand(-1, -1, events, 1)).squeeze(-1)
        if hard_p is not None:
            weights = _safe(hard_p).clamp_min(0) * event_valid[:, None]
        else:
            # A query-specific, predicted-only weighting when no assignment is supplied.
            compatibility = torch.maximum(lp_z, lp_second).masked_fill(~event_valid[:, None], -1e4)
            weights = compatibility.softmax(-1) * event_valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        values.extend((
            (weights * event_entropy[:, None]).sum(-1),
            (weights * ((lp_z - lp_second).abs() / 4).tanh()).sum(-1),
            (weights * ((lp_b - lp_z) / 4).tanh()).sum(-1),
        ))
    features = torch.stack(values, -1)
    if feature_mode == "no_mass_input":
        features = features.clone()
        features[..., list(MASS_FEATURE_INDICES)] = 0
    return torch.where(active[..., None], _safe(features), torch.zeros_like(features))


class BenefitRouter(nn.Module):
    """An initially neutral scalar router, trained from the first optimizer step."""

    def __init__(self, num_classes: int = 28, hidden_dim: int = 32,
                 feature_mode: str = "full", gate_upper: float = 2.0):
        super().__init__()
        if num_classes < 1 or hidden_dim < 2:
            raise ValueError("num_classes must be positive and hidden_dim at least two.")
        if feature_mode not in {"full", "no_mass_input"}:
            raise ValueError("feature_mode must be 'full' or 'no_mass_input'.")
        if gate_upper != 2.0:
            raise ValueError("R1 uses the fixed [0,2] gate interval; gate_upper must be 2.0.")
        self.num_classes = int(num_classes)
        self.feature_mode = feature_mode
        self.gate_upper = float(gate_upper)
        self.feature_names = FEATURE_NAMES
        self.mlp = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max(2, hidden_dim // 2)), nn.SiLU(),
            nn.Linear(max(2, hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, visual_logits: Tensor, base_logits: Tensor,
        candidate_mask: Tensor, transaction_gate: Tensor, *,
        scene_evidence: Optional[Tensor] = None,
        event_log_probability: Optional[Tensor] = None,
        weight_tokens_raw: Optional[Tensor] = None,
        weight_mask: Optional[Tensor] = None,
        query_record_residual: Optional[Tensor] = None,
        hard_p: Optional[Tensor] = None,
        row_margin: Optional[Tensor] = None,
        col_margin: Optional[Tensor] = None,
        constant_gate: Optional[float] = None,
        return_details: bool = False,
    ):
        if visual_logits.ndim != 3 or visual_logits.shape[-1] != self.num_classes:
            raise ValueError(f"visual_logits must have shape [B,Q,{self.num_classes}].")
        if constant_gate is not None and not 0 <= constant_gate <= self.gate_upper:
            raise ValueError("constant_gate must be between zero and two.")
        features = extract_router_features(
            visual_logits, base_logits, candidate_mask, transaction_gate,
            scene_evidence=scene_evidence, event_log_probability=event_log_probability,
            weight_tokens_raw=weight_tokens_raw, weight_mask=weight_mask,
            query_record_residual=query_record_residual, hard_p=hard_p,
            row_margin=row_margin, col_margin=col_margin, feature_mode=self.feature_mode,
        )
        active = candidate_mask.bool() & transaction_gate.bool()[:, None]
        visual_valid = torch.isfinite(visual_logits).all(-1)
        if (active & ~visual_valid).any():
            raise ValueError("Active queries require finite visual logits.")
        active &= visual_valid & torch.isfinite(base_logits).all(-1)
        visual = torch.where(torch.isfinite(visual_logits), visual_logits, torch.zeros_like(visual_logits))
        proposal = torch.where(torch.isfinite(base_logits), base_logits, visual)
        if constant_gate is None:
            learned_gate = self.gate_upper * torch.sigmoid(self.mlp(features).squeeze(-1))
            # Algebraically z+g*(base-z), arranged to give bitwise base at g=1
            # without a branch that would remove the initial gate gradient.
            routed = proposal + (learned_gate.to(proposal.dtype)[..., None] - 1) * (proposal - visual)
        else:
            learned_gate = features.new_full(features.shape[:-1], float(constant_gate))
            if constant_gate == 0:
                routed = visual
            elif constant_gate == 1:
                routed = proposal
            else:
                routed = proposal + (float(constant_gate) - 1) * (proposal - visual)
        logits = torch.where(active[..., None], routed, visual)
        effective_gate = torch.where(active, learned_gate, torch.zeros_like(learned_gate))
        if return_details:
            return {"pred_logits": logits, "gate": effective_gate,
                    "features": features, "active_mask": active}
        return logits
