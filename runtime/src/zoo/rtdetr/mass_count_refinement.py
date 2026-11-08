"""Inference-time mass-supported class multiplicity refinement.

For each class, independent event likelihoods induce a Poisson-binomial
distribution over the number of supporting records.  Its conditional survival
probability discounts the second and subsequent candidate queries for that
class.  The highest-scoring candidate is deliberately unchanged.  This is a
soft score correction, not a rule restricting a class to one object.

The module keeps the detector's native sigmoid-logit representation.  It does
not use annotations, record order, boxes, or learned parameters.  Sorting is
stable: exact ties retain the original query order.  Gradients exist conditional
on that discrete ordering; this module is intended primarily for inference.
"""

import math

import torch
from torch import nn


class MassCountLikelihoodRefinement(nn.Module):
    def __init__(self, strength=0.0, temperature=1.0, clip=6.0, prob_floor=1e-8):
        super().__init__()
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.clip = float(clip)
        self.prob_floor = float(prob_floor)
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be finite and positive.")
        if not math.isfinite(self.prob_floor) or not 0 < self.prob_floor <= 1:
            raise ValueError("prob_floor must lie in (0, 1].")

    @staticmethod
    def _validate_shapes(logits, log_probs, event_mask, candidate_mask, gate):
        if logits.ndim != 3 or not logits.is_floating_point() or logits.shape[-1] == 0:
            raise ValueError("refined_logits must be floating point [B, Q, C], with C > 0.")
        if log_probs.ndim != 3 or not log_probs.is_floating_point():
            raise ValueError("event_log_probs must be floating point [B, E, C].")
        if log_probs.shape[0] != logits.shape[0] or log_probs.shape[2] != logits.shape[2]:
            raise ValueError("event_log_probs batch and class dimensions must match logits.")
        if event_mask.shape != log_probs.shape[:2]:
            raise ValueError("event_mask must have shape [B, E].")
        if candidate_mask.shape != logits.shape[:2]:
            raise ValueError("candidate_mask must have shape [B, Q].")
        if gate.shape != logits.shape[:1]:
            raise ValueError("integrity_gate must have shape [B].")
        for name, mask in (("event_mask", event_mask), ("candidate_mask", candidate_mask),
                           ("integrity_gate", gate)):
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must have boolean dtype.")

    @staticmethod
    def _survival(probability):
        """Return P(N_c >= k), including k=0, for each batch/class.

        Each masked event has probability zero.  The dynamic program operates
        on the full padded event length, so tails above the usable event count
        are exactly zero.  Record permutations change only round-off error.
        """
        batch, event_count, classes = probability.shape
        distribution = probability.new_zeros((batch, classes, event_count + 1))
        distribution[..., 0] = 1.0
        for event in range(event_count):
            current = probability[:, event].unsqueeze(-1)
            arrival = torch.cat((distribution.new_zeros((batch, classes, 1)),
                                 distribution[..., :-1] * current), dim=-1)
            distribution = distribution * (1.0 - current) + arrival
        tail = distribution.flip(-1).cumsum(-1).flip(-1).clamp(0.0, 1.0)
        # P(N >= 0) is exactly one, independent of floating-point DP rounding.
        return torch.cat((torch.ones_like(tail[..., :1]), tail[..., 1:]), dim=-1)

    def forward(self, refined_logits, event_log_probs, event_mask,
                candidate_mask, integrity_gate):
        self._validate_shapes(refined_logits, event_log_probs, event_mask,
                              candidate_mask, integrity_gate)
        device = refined_logits.device
        dtype = torch.float64 if event_log_probs.dtype == torch.float64 else torch.float32
        event_mask = event_mask.to(device=device)
        candidate_mask = candidate_mask.to(device=device)
        integrity_gate = integrity_gate.to(device=device)
        log_probs = event_log_probs.to(device=device, dtype=dtype)

        # Padded NaN/Inf must never affect valid records.  A real record may
        # legitimately contain -Inf for a zero-probability class, but requires
        # at least one finite class and may not contain NaN or +Inf.
        row_valid = (torch.isfinite(log_probs).any(-1)
                     & ~torch.isnan(log_probs).any(-1)
                     & ~torch.isposinf(log_probs).any(-1))
        valid = (~event_mask | row_valid).all(-1)
        usable = event_mask & row_valid
        safe_logits = torch.where(usable.unsqueeze(-1), log_probs,
                                  torch.zeros_like(log_probs))
        # Subtract the maximum before temperature division to avoid positive
        # overflow for extremely small temperatures and large finite logits.
        centered = safe_logits - safe_logits.amax(-1, keepdim=True)
        probability = torch.softmax(centered / self.temperature, dim=-1)
        probability = torch.where(usable.unsqueeze(-1), probability,
                                  torch.zeros_like(probability))
        finite_probability = torch.isfinite(probability).all(dim=(1, 2))
        valid = valid & finite_probability
        probability = torch.where(valid[:, None, None], probability,
                                  torch.zeros_like(probability))
        tail = self._survival(probability)

        delta = torch.zeros_like(refined_logits)
        rank = torch.zeros_like(refined_logits, dtype=torch.long)
        rank_tail = probability.new_ones(refined_logits.shape)
        applied = (integrity_gate & valid & event_mask.any(-1)
                   & candidate_mask.any(-1))
        if self.strength == 0.0:
            applied = torch.zeros_like(applied)

        event_count = log_probs.shape[1]
        for batch in range(refined_logits.shape[0]):
            if not bool(applied[batch].item()):
                continue
            candidates = candidate_mask[batch].nonzero(as_tuple=False).flatten()
            scores = refined_logits[batch].index_select(0, candidates)
            order = torch.argsort(scores, dim=0, descending=True, stable=True)
            ranks = torch.empty_like(order)
            values = torch.arange(1, candidates.numel() + 1, device=device).unsqueeze(-1)
            ranks.scatter_(0, order, values.expand_as(order))
            candidate_tail = tail[batch].gather(1, ranks.clamp_max(event_count).T).T
            candidate_tail = torch.where(ranks <= event_count, candidate_tail,
                                         torch.zeros_like(candidate_tail))
            denominator = tail[batch, :, 1].clamp_min(self.prob_floor)
            conditional = (candidate_tail / denominator).clamp(self.prob_floor, 1.0)
            correction = self.strength * conditional.log().clamp(-self.clip, 0.0)
            # The first candidate is protected even if P(N_c >= 1) underflows.
            correction = torch.where(ranks == 1, torch.zeros_like(correction), correction)
            delta[batch, candidates] = correction.to(dtype=refined_logits.dtype)
            rank[batch, candidates] = ranks
            rank_tail[batch, candidates] = candidate_tail

        # Preserve the original tensor and values exactly when disabled.  The
        # explicit selector also preserves rejected/noncandidate signed zeros.
        output = refined_logits
        if self.strength != 0.0 and bool(applied.any().item()):
            changed = applied[:, None, None] & candidate_mask[:, :, None]
            output = torch.where(changed, refined_logits + delta, refined_logits)
        return {
            "pred_logits": output,
            "mass_count_delta": delta,
            "mass_count_tail_probability": tail,
            "mass_count_rank_tail_probability": rank_tail,
            "mass_count_event_probability": probability,
            "mass_count_rank": rank,
            "mass_count_applied": applied,
            "mass_count_valid": valid,
        }
