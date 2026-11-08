"""Parameter-free state-neutral association over an upstream fixed candidate pool.

Candidate generation and physical transaction verification are upstream CVSR
operations.  This module accepts their predicted-only tensor outputs; it does
not acquire records, check conservation, select boxes, or change localization.
SciPy's Hungarian solver introduces a CPU synchronization and is not an ONNX
or fully device-native inference path.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from scipy.optimize import linear_sum_assignment


def _require_tensor(value, shape, name):
    if not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape):
        actual = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {actual}.')


def _require_bool(value, name):
    if value.dtype != torch.bool:
        raise ValueError(f'{name} must be a boolean tensor.')


class StateNeutralRefinement(nn.Module):
    """Associate using parent means, then retain a fixed visual state contrast.

    Inputs:
        visual_logits: [B, Q, C], all original visual queries (Q=300 in the run).
        candidate_indices: [B, K], ordered upstream query indices; integer type.
        candidate_mask: [B, K], validity of padded candidate slots, NOT [B, Q].
        scene_evidence: [B, C], original CVSR scene evidence after verification.
        event_residual: [B, E, C], original event-versus-scene log residuals.
        weight_mask: [B, E], valid event slots, in the original event-axis order.
        transaction_gate: [B], result of the complete upstream integrity check.

    Means are taken within each fixed taxonomy parent.  On accepted candidates,
    U_ie=logsumexp_c(mean_parent(z_i)_c + 2*scene_c + residual_ec).
    Matching uses every Hungarian edge, with no positive-margin rejection.
    Output classification is z-alpha*(z-mean_parent(z))+(2*scene+.2*hard@residual).

    alpha defaults to .5.  Values 0 and 1 are explicit diagnostic readout
    variants; they do not change the state-neutral matching rule.  All arithmetic
    runs in FP32, while pred_logits retains visual_logits.dtype.  Non-candidates
    and rejected/unavailable transactions are copied from the raw tensor, without
    a float32 round trip.  Invalid active evidence also falls back transaction
    by transaction, but this numeric guard does not replace physical validation.

    Returned assignment and margins are indexed by full query and event axes.
    Unassigned/invalid rows have zero hard assignments, residuals and margins,
    and assigned_event=-1.  Signed row/column margins use zero when alternatives
    are absent, following the research-cache convention.  Equal-utility optima
    need not have unique assignments under query/event permutations.
    """

    def __init__(self, leaf_to_parent, alpha=.5):
        super().__init__()
        mapping = torch.as_tensor(leaf_to_parent)
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if mapping.ndim != 1 or not mapping.numel() or mapping.dtype not in integer_types:
            raise ValueError('leaf_to_parent must be a nonempty one-dimensional integer mapping.')
        mapping = mapping.detach().to(device='cpu', dtype=torch.long).clone()
        if (mapping < 0).any():
            raise ValueError('Parent indices must be nonnegative.')
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be finite and in [0, 1].')
        self.register_buffer('leaf_to_parent', mapping)
        self.register_buffer('readout_alpha', torch.tensor(alpha, dtype=torch.float32))

    @property
    def num_classes(self):
        return self.leaf_to_parent.numel()

    def _validate(self, visual_logits, candidate_indices, candidate_mask,
                  scene_evidence, event_residual, weight_mask, transaction_gate):
        if not isinstance(visual_logits, Tensor) or visual_logits.ndim != 3 or not visual_logits.is_floating_point():
            raise ValueError('visual_logits must be a floating [B, Q, C] tensor.')
        b, q, c = visual_logits.shape
        if q < 1 or c != self.num_classes:
            raise ValueError('The query axis must be nonempty and classes must match the taxonomy.')
        if not isinstance(candidate_indices, Tensor) or candidate_indices.ndim != 2:
            raise ValueError('candidate_indices must have shape [B, K].')
        k = candidate_indices.shape[1]
        _require_tensor(candidate_indices, (b, k), 'candidate_indices')
        if candidate_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('candidate_indices must have int32 or int64 dtype.')
        _require_tensor(candidate_mask, (b, k), 'candidate_mask')
        _require_bool(candidate_mask, 'candidate_mask')
        _require_tensor(scene_evidence, (b, c), 'scene_evidence')
        if not scene_evidence.is_floating_point():
            raise ValueError('scene_evidence must be floating point.')
        if not isinstance(event_residual, Tensor) or event_residual.ndim != 3:
            raise ValueError('event_residual must have shape [B, E, C].')
        e = event_residual.shape[1]
        _require_tensor(event_residual, (b, e, c), 'event_residual')
        if not event_residual.is_floating_point():
            raise ValueError('event_residual must be floating point.')
        _require_tensor(weight_mask, (b, e), 'weight_mask')
        _require_bool(weight_mask, 'weight_mask')
        _require_tensor(transaction_gate, (b,), 'transaction_gate')
        _require_bool(transaction_gate, 'transaction_gate')

    @torch.no_grad()
    def forward(self, visual_logits: Tensor, candidate_indices: Tensor,
                candidate_mask: Tensor, scene_evidence: Tensor,
                event_residual: Tensor, weight_mask: Tensor,
                transaction_gate: Tensor):
        self._validate(visual_logits, candidate_indices, candidate_mask,
                       scene_evidence, event_residual, weight_mask, transaction_gate)
        with torch.autocast(device_type=visual_logits.device.type, enabled=False):
            return self._forward_float32(visual_logits, candidate_indices, candidate_mask,
                                         scene_evidence, event_residual, weight_mask, transaction_gate)

    def _forward_float32(self, raw, candidate_indices, candidate_mask,
                         scene_evidence, event_residual, weight_mask, transaction_gate):
        device = raw.device
        b, q, c = raw.shape
        e = event_residual.shape[1]
        z = raw.float()
        indices = candidate_indices.to(device=device, dtype=torch.long)
        candidates = candidate_mask.to(device=device)
        scene = scene_evidence.to(device=device, dtype=torch.float32)
        residual = event_residual.to(device=device, dtype=torch.float32)
        valid_events = weight_mask.to(device=device)
        requested = transaction_gate.to(device=device)
        mapping = self.leaf_to_parent.to(device=device)
        alpha = self.readout_alpha.to(device=device, dtype=torch.float32)
        result = dict(
            pred_logits=raw.clone(),
            hard_assignment=torch.zeros((b, q, e), device=device, dtype=torch.float32),
            row_margin=torch.zeros((b, q), device=device, dtype=torch.float32),
            col_margin=torch.zeros((b, q), device=device, dtype=torch.float32),
            query_record_residual=torch.zeros((b, q, c), device=device, dtype=torch.float32),
            assigned_event=torch.full((b, q), -1, device=device, dtype=torch.long),
            applied_mask=torch.zeros((b, q), device=device, dtype=torch.bool),
            applied_transaction=torch.zeros(b, device=device, dtype=torch.bool),
            tensor_numeric_valid=torch.ones(b, device=device, dtype=torch.bool),
        )
        for batch in range(b):
            if not bool(requested[batch]):
                continue
            events = valid_events[batch].nonzero(as_tuple=False).flatten()
            slots = candidates[batch].nonzero(as_tuple=False).flatten()
            if not len(events) or not len(slots):
                continue
            query_ids = indices[batch, slots]
            if ((query_ids < 0) | (query_ids >= q)).any():
                raise ValueError('An active candidate index is outside the visual query axis.')
            if query_ids.unique().numel() != query_ids.numel():
                raise ValueError('An active candidate index occurs more than once.')
            original = z[batch, query_ids]
            event = residual[batch, events]
            if not (torch.isfinite(original).all() and torch.isfinite(scene[batch]).all()
                    and torch.isfinite(event).all()):
                result['tensor_numeric_valid'][batch] = False
                continue
            mean = original.clone()
            for parent in mapping.unique(sorted=True):
                leaves = (mapping == parent).nonzero(as_tuple=False).flatten()
                if len(leaves) > 1:
                    mean[:, leaves] = original[:, leaves].mean(-1, keepdim=True)
            utility = torch.logsumexp((mean + 2 * scene[batch])[:, None, :] + event[None, :, :], -1)
            if not torch.isfinite(utility).all():
                result['tensor_numeric_valid'][batch] = False
                continue
            rows_np, columns_np = linear_sum_assignment(-utility.detach().cpu().numpy())
            rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
            columns = torch.as_tensor(columns_np, device=device, dtype=torch.long)
            hard = torch.zeros((len(query_ids), len(events)), device=device, dtype=torch.float32)
            hard[rows, columns] = 1.
            # One event per matched query: indexed copying is exactly the
            # one-hot hard@event operation and avoids TF32 matrix-multiply
            # rounding if a caller has enabled TF32 globally on CUDA.
            proposal = torch.zeros((len(query_ids), c), device=device, dtype=torch.float32)
            proposal[rows] = event[columns]
            refined = original - alpha * (original - mean) + (2 * scene[batch] + .2 * proposal)
            cast = refined.to(dtype=raw.dtype)
            if not (torch.isfinite(refined).all() and torch.isfinite(cast).all()):
                result['tensor_numeric_valid'][batch] = False
                continue
            result['pred_logits'][batch, query_ids] = cast
            result['query_record_residual'][batch, query_ids] = proposal
            result['hard_assignment'][batch, query_ids[rows], events[columns]] = 1.
            result['assigned_event'][batch, query_ids[rows]] = events[columns]
            for row, column in zip(rows.tolist(), columns.tolist()):
                value = utility[row, column]
                if len(events) > 1:
                    other = torch.cat((utility[row, :column], utility[row, column + 1:]))
                    result['row_margin'][batch, query_ids[row]] = value - other.max()
                if len(query_ids) > 1:
                    other = torch.cat((utility[:row, column], utility[row + 1:, column]))
                    result['col_margin'][batch, query_ids[row]] = value - other.max()
            result['applied_mask'][batch, query_ids] = True
            result['applied_transaction'][batch] = True
        return result
