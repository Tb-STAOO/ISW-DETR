"""Confusion-factorized classification queries for RT-DETRv2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .cvsr import ConservationVerifiedSetRefinement
from .denoising import get_contrastive_denoising_training_group
from .rtdetr import RTDETR
from .rtdetrv2_decoder import RTDETRTransformerv2
from .utils import inverse_sigmoid


__all__ = ["CFQDRTDETRTransformerv2", "CFQDCVSRRTDETR"]


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-8)))


def _load_json(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _taxonomy_tensors(path, num_classes, num_states=2):
    raw = _load_json(path, "CFQD taxonomy")
    if int(raw.get("num_classes", -1)) != int(num_classes):
        raise ValueError("CFQD taxonomy class count does not match the model.")
    leaf_to_parent = torch.as_tensor(raw["leaf_to_parent"], dtype=torch.long)
    leaf_to_state = torch.as_tensor(
        raw.get("leaf_to_state", [-1] * num_classes), dtype=torch.long
    )
    if leaf_to_parent.numel() != num_classes or leaf_to_state.numel() != num_classes:
        raise ValueError("CFQD taxonomy must map every leaf class exactly once.")
    num_parents = int(raw.get("num_parents", leaf_to_parent.max().item() + 1))
    if leaf_to_parent.min().item() < 0 or leaf_to_parent.max().item() >= num_parents:
        raise ValueError("CFQD taxonomy contains an invalid parent index.")
    if "num_states" in raw and raw["num_states"] != num_states:
        raise ValueError("CFQD taxonomy num_states does not match cfqd_num_states.")
    if not torch.all((leaf_to_state >= -1) & (leaf_to_state < num_states)):
        raise ValueError(f"CFQD state labels must be -1 or in [0, {num_states - 1}].")

    counts = torch.as_tensor(raw.get("leaf_counts", [1] * num_classes), dtype=torch.float32)
    if counts.numel() != num_classes:
        raise ValueError("CFQD taxonomy leaf_counts has an invalid length.")
    balance = counts.clamp_min(1.0).rsqrt()
    balance = balance / balance.mean().clamp_min(1e-8)
    return raw, leaf_to_parent, leaf_to_state, balance, num_parents


def _confusion_candidates(
    taxonomy,
    leaf_to_parent,
    graph_path=None,
    include_parent_siblings=True,
    max_candidates=32,
):
    num_classes = int(leaf_to_parent.numel())
    candidates = [set() for _ in range(num_classes)]
    if include_parent_siblings:
        for first in range(num_classes):
            for second in range(num_classes):
                if first != second and leaf_to_parent[first] == leaf_to_parent[second]:
                    candidates[first].add(second)

    mapping = taxonomy.get("confusion_candidates", {})
    for key, values in mapping.items():
        label = int(key)
        if 0 <= label < num_classes:
            candidates[label].update(
                int(value)
                for value in values
                if 0 <= int(value) < num_classes and int(value) != label
            )

    if graph_path:
        graph = _load_json(graph_path, "CFQD confusion graph")
        if int(graph.get("num_classes", num_classes)) != num_classes:
            raise ValueError("CFQD confusion graph class count does not match the model.")
        for edge in graph.get("edges", []):
            classes = edge.get("classes", edge) if isinstance(edge, dict) else edge
            if len(classes) != 2:
                continue
            first, second = (int(value) for value in classes)
            if 0 <= first < num_classes and 0 <= second < num_classes and first != second:
                candidates[first].add(second)
                candidates[second].add(first)

    width = max(1, min(max_candidates, max((len(values) for values in candidates), default=1)))
    result = torch.full((num_classes, width), -1, dtype=torch.long)
    for label, values in enumerate(candidates):
        values = sorted(values)[:width]
        if values:
            result[label, : len(values)] = torch.as_tensor(values, dtype=torch.long)
    return result


class CFQDAdapter(nn.Module):
    def __init__(self, hidden_dim=256, adapter_dim=64, init_gamma=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return features + self.gamma * residual


class IdentityConditionedStateAttention(nn.Module):
    """Refine paired states from identity-conditioned local visual evidence."""

    def __init__(
        self,
        hidden_dim,
        num_parents,
        attention_dim=64,
        num_heads=4,
        grid_size=3,
        levels=(0, 1),
        box_expand=1.15,
        scale_init=0.01,
        max_scale=1.0,
        residual_limit=2.0,
        attention_type="icsa",
        use_identity_condition=True,
        use_null_evidence=True,
        use_signed_evidence=True,
        use_box_local_tokens=True,
        pair_constrained=True,
        trainable_scale=True,
    ):
        super().__init__()
        attention_dim = int(attention_dim)
        num_heads = int(num_heads)
        if attention_dim % num_heads != 0:
            raise ValueError("State-attention dimension must be divisible by num_heads.")
        if int(grid_size) < 2:
            raise ValueError("State-attention grid_size must be at least 2.")
        if attention_type not in {"mlp", "icsa", "ictea", "ictea_v2"}:
            raise ValueError(f"Unsupported state-attention type: {attention_type}")
        self.grid_size = int(grid_size)
        self.levels = tuple(int(level) for level in levels)
        self.box_expand = float(box_expand)
        self.max_scale = float(max_scale)
        self.residual_limit = float(residual_limit)
        self.attention_type = str(attention_type)
        self.use_identity_condition = bool(use_identity_condition)
        self.use_null_evidence = bool(use_null_evidence)
        self.use_signed_evidence = bool(use_signed_evidence)
        self.use_box_local_tokens = bool(use_box_local_tokens)
        self.pair_constrained = bool(pair_constrained)
        self.trainable_scale = bool(trainable_scale)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.identity_embedding = nn.Embedding(num_parents, attention_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_proj = nn.Linear(hidden_dim, attention_dim)
        self.cross_attention = None
        self.value_proj = None
        self.evidence_proj = None
        self.hypothesis_embedding = None
        if self.attention_type in {"mlp", "icsa"}:
            self.cross_attention = nn.MultiheadAttention(
                attention_dim, num_heads, dropout=0.0, batch_first=True
            )
        else:
            self.value_proj = nn.Linear(hidden_dim, attention_dim)
            self.evidence_proj = nn.Linear(attention_dim, attention_dim)
            hypothesis_count = 3 if self.use_null_evidence else 2
            self.hypothesis_embedding = nn.Embedding(hypothesis_count, attention_dim)
        self.output_norm = nn.LayerNorm(attention_dim)
        output_dim = 1 if self.pair_constrained else 2
        self.delta_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, output_dim),
        )
        self.scale_raw = nn.Parameter(torch.tensor(_inverse_softplus(scale_init)))

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        nn.init.normal_(self.identity_embedding.weight, std=0.02)
        if self.value_proj is not None:
            nn.init.xavier_uniform_(self.value_proj.weight)
            nn.init.zeros_(self.value_proj.bias)
            nn.init.xavier_uniform_(self.evidence_proj.weight)
            nn.init.zeros_(self.evidence_proj.bias)
            nn.init.normal_(self.hypothesis_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.delta_head[0].weight)
        nn.init.zeros_(self.delta_head[0].bias)
        nn.init.xavier_uniform_(self.delta_head[2].weight)
        nn.init.zeros_(self.delta_head[2].bias)

    @property
    def scale(self):
        return F.softplus(self.scale_raw).clamp(max=self.max_scale)

    def freeze_disabled_parameters(self):
        if not self.use_identity_condition:
            self.identity_embedding.requires_grad_(False)
        if not self.trainable_scale:
            self.scale_raw.requires_grad_(False)

    def _sample_local_tokens(self, memory, spatial_shapes, boxes):
        batch_size, query_count = boxes.shape[:2]
        if self.use_box_local_tokens:
            offsets = torch.linspace(
                -0.5,
                0.5,
                self.grid_size,
                dtype=boxes.dtype,
                device=boxes.device,
            )
            offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
            offset_grid = torch.stack((offset_x, offset_y), dim=-1).reshape(
                1, 1, -1, 2
            )
            centers = boxes[..., :2].unsqueeze(2)
            sizes = boxes[..., 2:].clamp_min(1e-4).unsqueeze(2) * self.box_expand
            sample_grid = (centers + offset_grid * sizes).clamp(0.0, 1.0)
        else:
            coordinates = torch.linspace(
                0.0,
                1.0,
                self.grid_size,
                dtype=boxes.dtype,
                device=boxes.device,
            )
            grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
            sample_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2)
            sample_grid = sample_grid.expand(batch_size, query_count, -1, -1)
        sample_grid = sample_grid.mul(2.0).sub(1.0)

        level_tokens = []
        start = 0
        selected_levels = set(self.levels)
        for level, (height, width) in enumerate(spatial_shapes):
            length = int(height) * int(width)
            if level in selected_levels:
                feature = memory[:, start : start + length]
                feature = feature.transpose(1, 2).reshape(
                    batch_size, memory.shape[-1], int(height), int(width)
                )
                sampled = F.grid_sample(
                    feature,
                    sample_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                sampled = sampled.permute(0, 2, 3, 1)
                level_tokens.append(sampled)
            start += length
        if not level_tokens:
            raise ValueError("State attention did not select a valid feature level.")
        return torch.cat(level_tokens, dim=2).reshape(
            batch_size * query_count, -1, memory.shape[-1]
        )

    def _mlp_control(self, query, local_tokens):
        """Use every MHA projection without content-dependent token weighting."""
        weight_q, weight_k, weight_v = self.cross_attention.in_proj_weight.chunk(3)
        if self.cross_attention.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = self.cross_attention.in_proj_bias.chunk(3)
        query_value = F.linear(query, weight_q, bias_q)
        key_value = F.linear(local_tokens, weight_k, bias_k).mean(dim=1, keepdim=True)
        token_value = F.linear(local_tokens, weight_v, bias_v).mean(dim=1, keepdim=True)
        fused = F.gelu(query_value + key_value + token_value)
        return self.cross_attention.out_proj(fused)

    def _ternary_evidence_attention(self, query, raw_tokens, key_tokens):
        batch_query, token_count = key_tokens.shape[:2]
        hypothesis_count = self.hypothesis_embedding.num_embeddings
        query_heads = query.reshape(batch_query, self.num_heads, self.head_dim)
        hypothesis_basis = self.hypothesis_embedding.weight.reshape(
            hypothesis_count, self.num_heads, self.head_dim
        ).permute(1, 0, 2)
        if self.attention_type == "ictea_v2":
            # Query-conditioned channel modulation makes the identity/query term
            # hypothesis-specific, so it cannot cancel in the hypothesis softmax.
            query_gate = 1.0 + torch.tanh(query_heads).unsqueeze(2)
            hypothesis = hypothesis_basis.unsqueeze(0) * query_gate
        else:
            hypothesis = query_heads.unsqueeze(2) + hypothesis_basis.unsqueeze(0)
        key_heads = key_tokens.reshape(
            batch_query, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        scores = torch.einsum("bhrd,bhkd->bhkr", hypothesis, key_heads)
        scores = scores / math.sqrt(self.head_dim)
        routing = scores.softmax(dim=-1)

        empty_probability = routing[..., 0]
        full_probability = routing[..., 1]
        if self.use_signed_evidence:
            token_weight = full_probability - empty_probability
        else:
            token_weight = full_probability + empty_probability
        normalizer = token_weight.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        token_weight = token_weight / normalizer

        value_tokens = self.value_proj(self.token_norm(raw_tokens))
        value_heads = value_tokens.reshape(
            batch_query, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        evidence = torch.einsum("bhk,bhkd->bhd", token_weight, value_heads)
        evidence = evidence.reshape(batch_query, self.attention_dim)
        evidence = self.evidence_proj(evidence).unsqueeze(1)

        routing_summary = routing.mean(dim=(1, 2))
        if hypothesis_count == 2:
            routing_summary = F.pad(routing_summary, (0, 1))
        return evidence, routing_summary

    def forward(
        self,
        query_features,
        parent_logits,
        memory,
        spatial_shapes,
        boxes,
        paired_parent_mask,
    ):
        query_features = query_features.detach()
        parent_probabilities = parent_logits.detach().softmax(dim=-1)
        memory = memory.detach()
        boxes = boxes.detach()

        if self.use_identity_condition:
            identity_context = parent_probabilities @ self.identity_embedding.weight
        else:
            identity_context = torch.zeros(
                *parent_probabilities.shape[:-1],
                self.attention_dim,
                dtype=parent_probabilities.dtype,
                device=parent_probabilities.device,
            )
        query = self.query_proj(self.query_norm(query_features)) + identity_context
        raw_tokens = self._sample_local_tokens(memory, spatial_shapes, boxes)
        local_tokens = self.token_proj(self.token_norm(raw_tokens))

        batch_size, query_count = query.shape[:2]
        query = query.reshape(batch_size * query_count, 1, -1)
        routing_summary = None
        if self.attention_type == "icsa":
            attended, _ = self.cross_attention(
                query, local_tokens, local_tokens, need_weights=False
            )
        elif self.attention_type == "mlp":
            attended = self._mlp_control(query, local_tokens)
        else:
            attended, routing_summary = self._ternary_evidence_attention(
                query.squeeze(1), raw_tokens, local_tokens
            )
        attended = self.output_norm(query + attended)
        residual = self.delta_head(attended).reshape(
            batch_size, query_count, -1
        )
        residual = torch.tanh(residual) * self.residual_limit
        if self.pair_constrained:
            delta = residual.squeeze(-1)
            state_residual = torch.stack((-delta, delta), dim=-1)
        else:
            state_residual = residual
            delta = 0.5 * (state_residual[..., 1] - state_residual[..., 0])
        paired_probability = parent_probabilities @ paired_parent_mask.to(
            dtype=parent_probabilities.dtype
        )
        state_residual = state_residual * paired_probability.unsqueeze(-1)
        delta = delta * paired_probability
        if routing_summary is not None:
            routing_summary = routing_summary.reshape(batch_size, query_count, 3)
        return state_residual, delta, routing_summary


@register()
class CFQDRTDETRTransformerv2(RTDETRTransformerv2):
    """RT-DETRv2 decoder with a reference-tied classification query stream."""

    def __init__(
        self,
        *args,
        taxonomy_path,
        cfqd_variant="full",
        cfqd_num_states=2,
        cfqd_branch_start=1,
        cfqd_adapter_dim=64,
        cfqd_init_gamma=0.0,
        cfqd_factor_scale_init=0.01,
        cfqd_confusion_graph_path=None,
        cfqd_include_parent_siblings=True,
        cfqd_max_confusion_candidates=32,
        cfqd_enable_state_attention=False,
        cfqd_state_attention_dim=64,
        cfqd_state_attention_heads=4,
        cfqd_state_attention_grid_size=3,
        cfqd_state_attention_levels=(0, 1),
        cfqd_state_attention_box_expand=1.15,
        cfqd_state_attention_scale_init=0.01,
        cfqd_state_attention_max_scale=1.0,
        cfqd_state_attention_residual_limit=2.0,
        cfqd_state_attention_type="icsa",
        cfqd_state_attention_use_identity=True,
        cfqd_state_attention_use_null=True,
        cfqd_state_attention_use_signed=True,
        cfqd_state_attention_use_box_local=True,
        cfqd_state_attention_pair_constrained=True,
        cfqd_state_attention_trainable_scale=True,
        cfqd_attention_only_training=False,
        **kwargs,
    ):
        kwargs["expose_query_features"] = True
        super().__init__(*args, **kwargs)
        if cfqd_variant not in {"dq_only", "bhcl", "factorized", "full"}:
            raise ValueError(f"Unsupported CFQD variant: {cfqd_variant}")
        if not 0 <= int(cfqd_branch_start) < self.num_layers:
            raise ValueError("cfqd_branch_start must select a decoder layer.")
        self.cfqd_variant = cfqd_variant
        self.cfqd_branch_start = int(cfqd_branch_start)
        self.cfqd_factorized = cfqd_variant in {"factorized", "full"}
        if isinstance(cfqd_num_states, bool) or not isinstance(cfqd_num_states, int) or cfqd_num_states < 1:
            raise ValueError("cfqd_num_states must be a positive integer.")
        self.cfqd_num_states = cfqd_num_states
        self.cfqd_enable_state_attention = bool(cfqd_enable_state_attention)
        self.cfqd_attention_only_training = bool(cfqd_attention_only_training)
        if self.cfqd_enable_state_attention and not self.cfqd_factorized:
            raise ValueError("State attention requires a factorized CFQD variant.")
        if self.cfqd_enable_state_attention and self.cfqd_num_states != 2:
            raise ValueError("The optional state-attention module is binary; disable it for multistate I-S.")
        taxonomy, leaf_to_parent, leaf_to_state, balance, num_parents = _taxonomy_tensors(
            taxonomy_path, self.num_classes, self.cfqd_num_states
        )
        self.num_parents = num_parents
        self.register_buffer("cfqd_leaf_to_parent", leaf_to_parent, persistent=True)
        self.register_buffer("cfqd_leaf_to_state", leaf_to_state, persistent=True)
        self.register_buffer("cfqd_leaf_balance", balance, persistent=True)
        self.register_buffer(
            "cfqd_confusion_candidates",
            _confusion_candidates(
                taxonomy,
                leaf_to_parent,
                graph_path=cfqd_confusion_graph_path,
                include_parent_siblings=cfqd_include_parent_siblings,
                max_candidates=int(cfqd_max_confusion_candidates),
            ),
            persistent=True,
        )

        self.cfqd_adapters = nn.ModuleList(
            [
                CFQDAdapter(
                    hidden_dim=self.hidden_dim,
                    adapter_dim=int(cfqd_adapter_dim),
                    init_gamma=float(cfqd_init_gamma),
                )
                for _ in range(self.num_layers - self.cfqd_branch_start)
            ]
        )
        self.cfqd_parent_head = nn.Linear(self.hidden_dim, self.num_parents)
        self.cfqd_state_head = nn.Linear(self.hidden_dim, self.cfqd_num_states)
        self.cfqd_parent_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(cfqd_factor_scale_init))
        )
        self.cfqd_state_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(cfqd_factor_scale_init))
        )
        nn.init.xavier_uniform_(self.cfqd_parent_head.weight)
        nn.init.zeros_(self.cfqd_parent_head.bias)
        nn.init.xavier_uniform_(self.cfqd_state_head.weight)
        nn.init.zeros_(self.cfqd_state_head.bias)
        paired_parent = torch.zeros(self.num_parents, dtype=torch.bool)
        for parent_index in range(self.num_parents):
            states = leaf_to_state[leaf_to_parent == parent_index]
            # The historical buffer name is retained for binary checkpoints.
            # For multistate taxonomies it denotes at least two valid states.
            paired_parent[parent_index] = states[states >= 0].unique().numel() > 1
        self.register_buffer("cfqd_paired_parent", paired_parent, persistent=True)
        self.cfqd_state_attention = None
        if self.cfqd_enable_state_attention:
            self.cfqd_state_attention = IdentityConditionedStateAttention(
                hidden_dim=self.hidden_dim,
                num_parents=self.num_parents,
                attention_dim=cfqd_state_attention_dim,
                num_heads=cfqd_state_attention_heads,
                grid_size=cfqd_state_attention_grid_size,
                levels=cfqd_state_attention_levels,
                box_expand=cfqd_state_attention_box_expand,
                scale_init=cfqd_state_attention_scale_init,
                max_scale=cfqd_state_attention_max_scale,
                residual_limit=cfqd_state_attention_residual_limit,
                attention_type=cfqd_state_attention_type,
                use_identity_condition=cfqd_state_attention_use_identity,
                use_null_evidence=cfqd_state_attention_use_null,
                use_signed_evidence=cfqd_state_attention_use_signed,
                use_box_local_tokens=cfqd_state_attention_use_box_local,
                pair_constrained=cfqd_state_attention_pair_constrained,
                trainable_scale=cfqd_state_attention_trainable_scale,
            )
        self._cfqd_base_frozen = False

    def freeze_base_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if not self.cfqd_attention_only_training:
            self.cfqd_adapters.requires_grad_(True)
        if self.cfqd_factorized and not self.cfqd_attention_only_training:
            self.cfqd_parent_head.requires_grad_(True)
            self.cfqd_state_head.requires_grad_(True)
            self.cfqd_parent_scale_raw.requires_grad_(True)
            self.cfqd_state_scale_raw.requires_grad_(True)
        if self.cfqd_state_attention is not None:
            self.cfqd_state_attention.requires_grad_(True)
            self.cfqd_state_attention.freeze_disabled_parameters()
        self._cfqd_base_frozen = True

    def _set_base_eval(self):
        for module in (
            self.input_proj,
            self.decoder.layers,
            self.query_pos_head,
            self.enc_output,
            self.enc_score_head,
            self.enc_bbox_head,
            self.dec_score_head,
            self.dec_bbox_head,
        ):
            module.eval()
        if hasattr(self, "denoising_class_embed"):
            self.denoising_class_embed.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self._cfqd_base_frozen:
            self._set_base_eval()
            if not self.cfqd_attention_only_training:
                self.cfqd_adapters.train(True)
                self.cfqd_parent_head.train(True)
                self.cfqd_state_head.train(True)
            if self.cfqd_state_attention is not None:
                self.cfqd_state_attention.train(True)
        return self

    def _decode_queries(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        attn_mask=None,
    ):
        base_boxes = []
        base_logits = []
        location_features = []
        classification_features = []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        location = target
        classification = target.detach()

        for index, layer in enumerate(self.decoder.layers):
            reference_input = ref_points_detach.unsqueeze(2)
            query_position = self.query_pos_head(ref_points_detach)
            location = layer(
                location,
                reference_input,
                memory,
                spatial_shapes,
                attn_mask,
                None,
                query_position,
            )
            location_features.append(location)

            if index < self.cfqd_branch_start:
                classification = location.detach()
            else:
                classification = layer(
                    classification,
                    reference_input.detach(),
                    memory.detach(),
                    spatial_shapes,
                    attn_mask,
                    None,
                    query_position.detach(),
                )
                classification = self.cfqd_adapters[
                    index - self.cfqd_branch_start
                ](classification)
            classification_features.append(classification)

            intermediate_box = torch.sigmoid(
                self.dec_bbox_head[index](location) + inverse_sigmoid(ref_points_detach)
            )
            if self.training:
                base_logits.append(self.dec_score_head[index](location))
                if index == 0:
                    base_boxes.append(intermediate_box)
                else:
                    base_boxes.append(
                        torch.sigmoid(
                            self.dec_bbox_head[index](location)
                            + inverse_sigmoid(ref_points)
                        )
                    )
            elif index == self.decoder.eval_idx:
                base_logits.append(self.dec_score_head[index](location))
                base_boxes.append(intermediate_box)
                break

            ref_points = intermediate_box
            ref_points_detach = intermediate_box.detach()

        return (
            torch.stack(base_boxes),
            torch.stack(base_logits),
            torch.stack(location_features),
            torch.stack(classification_features),
        )

    def _factorized_logits(
        self, classification_features, memory=None, spatial_shapes=None, boxes=None
    ):
        leaf_logits = self.dec_score_head[-1](classification_features)
        parent_logits = self.cfqd_parent_head(classification_features)
        base_state_logits = self.cfqd_state_head(classification_features)
        state_logits = base_state_logits
        state_attention_delta = None
        state_attention_residual = None
        state_attention_routing = None
        final_logits = leaf_logits
        if self.cfqd_factorized:
            parent_scale = F.softplus(self.cfqd_parent_scale_raw).clamp(max=1.0)
            state_scale = F.softplus(self.cfqd_state_scale_raw).clamp(max=1.0)
            final_logits = final_logits + parent_scale * parent_logits.index_select(
                -1, self.cfqd_leaf_to_parent
            )
            state_index = self.cfqd_leaf_to_state.clamp_min(0)
            expanded_state = base_state_logits.index_select(-1, state_index)
            paired = (self.cfqd_leaf_to_state >= 0).to(expanded_state.dtype)
            final_logits = final_logits + state_scale * expanded_state * paired
            if self.cfqd_state_attention is not None:
                if memory is None or spatial_shapes is None or boxes is None:
                    raise ValueError("State attention requires memory, shapes, and boxes.")
                (
                    state_attention_residual,
                    state_attention_delta,
                    state_attention_routing,
                ) = self.cfqd_state_attention(
                    classification_features,
                    parent_logits,
                    memory,
                    spatial_shapes,
                    boxes,
                    self.cfqd_paired_parent,
                )
                state_logits = base_state_logits + state_attention_residual
                attention_by_leaf = state_attention_residual.index_select(
                    -1, state_index
                )
                final_logits = final_logits + self.cfqd_state_attention.scale * (
                    attention_by_leaf * paired
                )
        return (
            final_logits,
            leaf_logits,
            parent_logits,
            state_logits,
            state_attention_delta,
            state_attention_residual,
            state_attention_routing,
        )

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
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
            )
        else:
            denoising_logits = denoising_bbox_unact = attn_mask = dn_meta = None

        content, reference_unact, encoder_boxes, encoder_logits = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )
        boxes, visual_logits, location_features, classification_features = (
            self._decode_queries(
                content,
                reference_unact,
                memory,
                spatial_shapes,
                attn_mask=attn_mask,
            )
        )

        if self.training and dn_meta is not None:
            dn_boxes, boxes = torch.split(boxes, dn_meta["dn_num_split"], dim=2)
            dn_logits, visual_logits = torch.split(
                visual_logits, dn_meta["dn_num_split"], dim=2
            )
            _, location_features = torch.split(
                location_features, dn_meta["dn_num_split"], dim=2
            )
            _, classification_features = torch.split(
                classification_features, dn_meta["dn_num_split"], dim=2
            )

        final_features = classification_features[-1]
        (
            final_logits,
            leaf_logits,
            parent_logits,
            state_logits,
            state_attention_delta,
            state_attention_residual,
            state_attention_routing,
        ) = self._factorized_logits(
            final_features,
            memory=memory,
            spatial_shapes=spatial_shapes,
            boxes=boxes[-1],
        )
        output = {
            "pred_logits": final_logits,
            "pred_boxes": boxes[-1],
            "pred_logits_base": visual_logits[-1],
            "pred_logits_matching": visual_logits[-1],
            "pred_leaf_residual_logits": leaf_logits,
            "pred_parent_logits": parent_logits,
            "pred_state_logits": state_logits,
            "pred_cls_features": final_features,
            "pred_query_features": location_features[-1],
            "class_prototypes": self.dec_score_head[-1].weight,
            "cfqd_leaf_to_parent": self.cfqd_leaf_to_parent,
            "cfqd_leaf_to_state": self.cfqd_leaf_to_state,
            "cfqd_leaf_balance": self.cfqd_leaf_balance,
            "cfqd_confusion_candidates": self.cfqd_confusion_candidates,
        }
        if state_attention_delta is not None:
            output["pred_state_attention_delta"] = state_attention_delta
            output["pred_state_attention_residual"] = state_attention_residual
            output["state_attention_scale"] = self.cfqd_state_attention.scale
            if state_attention_routing is not None:
                output["pred_state_attention_routing"] = state_attention_routing
        if self.training and self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                visual_logits[:-1], boxes[:-1]
            )
            output["enc_aux_outputs"] = self._set_aux_loss(
                encoder_logits, encoder_boxes
            )
            output["enc_meta"] = {
                "class_agnostic": self.query_select_method == "agnostic"
            }
            if dn_meta is not None:
                output["dn_aux_outputs"] = self._set_aux_loss(dn_logits, dn_boxes)
                output["dn_meta"] = dn_meta
        return output


@register()
class CFQDCVSRRTDETR(RTDETR):
    """RT-DETR wrapper combining CFQD visual logits with optional locked CVSR."""

    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["num_classes"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes=80,
        freeze_detector=True,
        enable_cvsr=False,
        physics_stats_path=None,
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
        cvsr_count_refinement_strength=0.0,
        cvsr_count_refinement_temperature=1.0,
        cvsr_ambiguity_adapter_checkpoint=None,
    ):
        super().__init__(backbone=backbone, encoder=encoder, decoder=decoder)
        if not isinstance(decoder, CFQDRTDETRTransformerv2):
            raise TypeError("CFQDCVSRRTDETR requires CFQDRTDETRTransformerv2.")
        self.freeze_detector = bool(freeze_detector)
        self.cvsr_apply_during_training = bool(cvsr_apply_during_training)
        if self.freeze_detector:
            self.backbone.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.freeze_base_parameters()

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
        if mode and self.freeze_detector:
            self.backbone.eval()
            self.encoder.eval()
            self.decoder._set_base_eval()
        return self

    def forward(self, x, targets=None, modal_inputs=None, visual_inputs=None):
        features = self.encoder(self.backbone(x))
        outputs = self.decoder(features, targets=targets)
        outputs["pred_logits_visual"] = outputs["pred_logits_base"]
        outputs["pred_logits_cfqd"] = outputs["pred_logits"]
        if self.cvsr is None or (self.training and not self.cvsr_apply_during_training):
            outputs["pred_logits_cvsr"] = outputs["pred_logits"]
            return outputs

        cvsr_outputs = self.cvsr(
            outputs["pred_logits"], outputs["pred_boxes"], modal_inputs
        )
        outputs["pred_logits"] = cvsr_outputs.pop("pred_logits")
        outputs["pred_logits_cvsr"] = outputs["pred_logits"]
        outputs["pred_logits_scene"] = cvsr_outputs.pop("pred_logits_scene")
        outputs.update({f"cvsr_{key}": value for key, value in cvsr_outputs.items()})
        return outputs
