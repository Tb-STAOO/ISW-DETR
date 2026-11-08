"""Frozen-CFQD local/context classification experiments; disabled by default.

HiLo-style splitting follows Pan et al., Fast Vision Transformers with HiLo
Attention (NeurIPS 2022). This independent implementation operates on spatial
ROI grids, never on the unordered detector-query sequence.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def position_encoding(xy, dim=64):
    frequencies = 10000 ** (-torch.arange(dim // 4, device=xy.device).float() / (dim // 4))
    phase = xy.float().unsqueeze(-1) * (2 * math.pi) * frequencies
    return torch.cat((phase[..., 0, :].sin(), phase[..., 0, :].cos(),
                      phase[..., 1, :].sin(), phase[..., 1, :].cos()), -1)


class SpatialMixer(nn.Module):
    def __init__(self, hilo=False, dim=64):
        super().__init__()
        self.hilo = hilo
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, dim))

    def forward(self, tokens):
        n, length, dim = tokens.shape
        if length != 64:
            raise ValueError('SpatialMixer requires an 8x8 ROI grid.')
        normalized = self.norm(tokens)
        q, k, v = self.qkv(normalized).reshape(n, 64, 3, 4, 16).permute(2, 0, 3, 1, 4)
        if not self.hilo:
            mixed = F.scaled_dot_product_attention(q, k, v)
        else:
            # Two heads attend within 2x2 windows; two use pooled ROI keys/values.
            def windows(t):
                return t[:, :2].reshape(n, 2, 4, 2, 4, 2, 16).permute(
                    0, 2, 4, 1, 3, 5, 6).reshape(n * 16, 2, 4, 16)
            high = F.scaled_dot_product_attention(windows(q), windows(k), windows(v))
            high = high.reshape(n, 4, 4, 2, 2, 2, 16).permute(
                0, 3, 1, 4, 2, 5, 6).reshape(n, 2, 64, 16)
            def pooled(t):
                t = t[:, 2:].transpose(-1, -2).reshape(n, 32, 8, 8)
                return F.avg_pool2d(t, 2).reshape(n, 2, 16, 16).transpose(-1, -2)
            low = F.scaled_dot_product_attention(q[:, 2:], pooled(k), pooled(v))
            mixed = torch.cat((high, low), 1)
        tokens = tokens + self.proj(mixed.transpose(1, 2).reshape(n, 64, dim))
        return tokens + self.ffn(self.norm2(tokens))


class VisualSidecar(nn.Module):
    def __init__(self, mode, hidden_dim=256, num_classes=28, num_parents=18):
        super().__init__()
        if mode not in ('C1', 'C2', 'C3', 'C4'):
            raise ValueError(mode)
        self.mode = mode
        self.query_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64))
        self.feature_proj = nn.ModuleList([nn.Conv2d(hidden_dim, 64, 1) for _ in range(2)])
        self.level_embed = nn.Parameter(torch.zeros(2, 64))
        self.mixer = SpatialMixer(hilo=mode in ('C3', 'C4')) if mode != 'C1' else None
        self.reader = nn.MultiheadAttention(64, 4, dropout=0, batch_first=True) if mode != 'C1' else None
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, count))
            for count in (num_classes, num_parents, 2)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # Instantiate gates last so C3/C4 share every common seeded weight.
        self.gate = nn.Linear(64, 3 if mode == 'C4' else 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, query, boxes, features):
        if query.ndim != 3 or boxes.shape != (*query.shape[:2], 4) or len(features) < 2:
            raise ValueError('Expected BxQxC queries, BxQx4 boxes and P3/P4 features.')
        if not torch.isfinite(boxes).all() or (boxes[..., 2:] < 0).any():
            raise ValueError('Invalid predicted ROI boxes.')
        b, count, _ = query.shape
        q = self.query_proj(query.detach()).reshape(b * count, 1, 64)
        offsets = torch.linspace(-0.5, 0.5, 8, device=boxes.device, dtype=boxes.dtype)
        yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
        grid = torch.stack((xx, yy), -1).reshape(1, 1, 64, 2)
        xy = (boxes.detach()[..., :2, None].transpose(-1, -2) +
              grid * boxes.detach()[..., 2:, None].transpose(-1, -2).clamp_min(1e-4) * 1.15).clamp(0, 1)
        local, context = [], []
        coordinates = (torch.arange(4, device=boxes.device).float() + 0.5) / 4
        gy, gx = torch.meshgrid(coordinates, coordinates, indexing='ij')
        context_pos = position_encoding(torch.stack((gx, gy), -1).reshape(16, 2))
        for level in range(2):
            feature = features[level]
            if feature.ndim != 4 or feature.shape[0] != b:
                raise ValueError('Invalid feature map dimensions.')
            projected = self.feature_proj[level](feature.detach())
            sampled = F.grid_sample(projected, (xy * 2 - 1).to(projected.dtype),
                                    mode='bilinear', padding_mode='border', align_corners=False)
            tokens = sampled.permute(0, 2, 3, 1).reshape(b * count, 64, 64)
            tokens = tokens + position_encoding(xy).reshape(b * count, 64, 64).to(tokens.dtype)
            tokens = tokens + self.level_embed[level]
            if self.mixer is not None:
                tokens = self.mixer(tokens)
            local.append(tokens)
            ctx = F.adaptive_avg_pool2d(projected, 4).flatten(2).transpose(1, 2)
            ctx = ctx + context_pos.to(ctx.dtype) + self.level_embed[level]
            context.append(ctx[:, None].expand(b, count, 16, 64).reshape(b * count, 16, 64))
        local, context = torch.cat(local, 1), torch.cat(context, 1)
        if self.reader is None:
            local, context = local.mean(1, keepdim=True), context.mean(1, keepdim=True)
        else:
            local = self.reader(q, local, local, need_weights=False)[0]
            context = self.reader(q, context, context, need_weights=False)[0]
        gates = self.gate(q).sigmoid()
        deltas = []
        for index, head in enumerate(self.heads):
            alpha = gates[..., index:index+1] if self.mode == 'C4' else gates
            evidence = alpha * local + (1 - alpha) * context
            raw = head(torch.cat((q, evidence), -1))
            deltas.append((2 * torch.tanh(raw / 2)).reshape(b, count, -1))
        return (*deltas, gates.reshape(b, count, -1))


class FrozenCFQDSidecar(nn.Module):
    def __init__(self, base, mode=None):
        super().__init__()
        if base.cvsr is not None:
            raise ValueError('Visual sidecar requires an unfused CFQD base.')
        self.base = base.requires_grad_(False).eval()
        self.sidecar = None if mode is None else VisualSidecar(mode, base.decoder.hidden_dim,
                                                              base.decoder.num_classes,
                                                              base.decoder.num_parents)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, images, targets=None):
        with torch.no_grad():
            features = self.base.encoder(self.base.backbone(images))
            original = self.base.decoder(features)
        result = dict(original)
        if self.sidecar is not None:
            leaf, parent, state, gates = self.sidecar(
                original['pred_cls_features'], original['pred_boxes'], features)
            decoder = self.base.decoder
            ps = F.softplus(decoder.cfqd_parent_scale_raw).clamp(max=1)
            ss = F.softplus(decoder.cfqd_state_scale_raw).clamp(max=1)
            delta = leaf + ps * parent.index_select(-1, decoder.cfqd_leaf_to_parent)
            delta = delta + ss * state.index_select(-1, decoder.cfqd_leaf_to_state.clamp_min(0)) * (
                decoder.cfqd_leaf_to_state >= 0).to(state.dtype)
            result['pred_logits'] = original['pred_logits'] + delta
            result['pred_parent_logits'] = original['pred_parent_logits'] + parent
            result['pred_state_logits'] = original['pred_state_logits'] + state
            result['pred_leaf_residual_logits'] = original['pred_leaf_residual_logits'] + leaf
            result['sidecar_gates'] = gates
        result['pred_logits_frozen_cfqd'] = original['pred_logits']
        result['pred_logits_visual'] = original['pred_logits_base']
        result['pred_logits_cfqd'] = result['pred_logits']
        result['pred_logits_cvsr'] = result['pred_logits']
        return result
