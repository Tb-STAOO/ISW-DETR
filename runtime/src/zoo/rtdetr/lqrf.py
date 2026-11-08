"""Learned query-record fusion baseline for ScaleVision-MM."""

import torch
import torch.nn as nn

from ...core import register
from .rtdetr import RTDETR
from .rtdetrv2_decoder import ScaleWeightTokenEncoder


@register()
class LQRFRTDETR(RTDETR):
    """Freeze RT-DETRv2 and learn a classification-only query-record residual.

    This baseline intentionally excludes analytic mass likelihoods, conservation
    checks, explicit assignment, candidate filtering, and event-index
    supervision. Missing records are masked structurally so the output is
    exactly the frozen visual prediction.
    """

    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["num_classes"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes=80,
        weight_input_dim=7,
        record_encoder_layers=2,
        record_encoder_heads=4,
        record_encoder_ffn_dim=512,
        fusion_heads=8,
        dropout=0.1,
        residual_hidden_dim=256,
        residual_init_gate=0.0,
        residual_max_gate=1.0,
        expose_attention=False,
        freeze_detector=True,
    ):
        super().__init__(backbone=backbone, encoder=encoder, decoder=decoder)
        self.freeze_detector = bool(freeze_detector)
        self.expose_attention = bool(expose_attention)
        self.residual_max_gate = float(residual_max_gate)

        hidden_dim = int(getattr(decoder, "hidden_dim", 256))
        self.record_encoder = ScaleWeightTokenEncoder(
            input_dim=int(weight_input_dim),
            hidden_dim=hidden_dim,
            num_layers=int(record_encoder_layers),
            nhead=int(record_encoder_heads),
            dim_feedforward=int(record_encoder_ffn_dim),
            dropout=float(dropout),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.record_norm = nn.LayerNorm(hidden_dim)
        self.query_record_attention = nn.MultiheadAttention(
            hidden_dim,
            int(fusion_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden_dim), int(num_classes)),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init_gate)))

        if self.freeze_detector:
            self.backbone.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_detector:
            self.backbone.eval()
            self.encoder.eval()
            self.decoder.eval()
        return self

    def _encode_records(self, modal_inputs, dtype, device):
        if modal_inputs is None:
            return None, None, None
        tokens = modal_inputs.get("weight_tokens")
        mask = modal_inputs.get("weight_mask")
        if tokens is None or mask is None:
            return None, None, None

        tokens = tokens.to(device=device, dtype=dtype)
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("LQRF requires [batch, events, features] and [batch, events] mask.")
        mask = mask.to(device=device, dtype=torch.bool)
        valid = (~mask | torch.isfinite(tokens).all(dim=-1)).all(dim=1)
        raw = modal_inputs.get('weight_tokens_raw')
        if raw is not None:
            if raw.shape != (*tokens.shape[:2], 7):
                raise ValueError("weight_tokens_raw must have shape [batch, events, 7].")
            raw = raw.to(device=device)
            statistics_valid = ((raw[..., 3:5] >= 0).all(dim=-1) & (raw[..., 6] >= 1)
                                & (raw[..., 6] == raw[..., 6].round()))
            valid &= (~mask | (torch.isfinite(raw).all(dim=-1) & statistics_valid)).all(dim=1)
        mask = mask & valid[:, None]
        tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        has_records = mask.any(dim=1)
        if tokens.shape[1] == 0:
            tokens = tokens.new_zeros((tokens.shape[0], 1, tokens.shape[-1]))
            mask = torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=device)

        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
        encoded = self.record_encoder(tokens, safe_mask)
        return encoded, safe_mask, has_records

    def forward(self, x, targets=None, modal_inputs=None, visual_inputs=None):
        del visual_inputs
        if self.freeze_detector:
            with torch.no_grad():
                features = self.encoder(self.backbone(x))
                outputs = self.decoder(features, targets=None)
        else:
            features = self.encoder(self.backbone(x))
            outputs = self.decoder(features, targets=targets)

        if "pred_query_features" not in outputs:
            raise KeyError("LQRF requires decoder expose_query_features=True.")

        visual_logits = outputs["pred_logits"]
        query_features = outputs["pred_query_features"].detach()
        encoded, safe_mask, has_records = self._encode_records(
            modal_inputs,
            dtype=query_features.dtype,
            device=query_features.device,
        )

        residual = torch.zeros_like(visual_logits)
        attention = None
        if encoded is not None:
            update, attention = self.query_record_attention(
                self.query_norm(query_features),
                self.record_norm(encoded),
                encoded,
                key_padding_mask=~safe_mask,
                need_weights=self.expose_attention,
                average_attn_weights=True,
            )
            residual = self.residual_head(self.residual_norm(update))
            residual = torch.where(has_records[:, None, None], residual, torch.zeros_like(residual))

        gate = self.residual_gate.clamp(
            min=-self.residual_max_gate,
            max=self.residual_max_gate,
        )
        refined_logits = visual_logits + gate * residual
        # Keep a trainable zero anchor for batches where every record is missing.
        refined_logits = refined_logits + self.residual_gate * 0.0

        outputs["pred_logits_visual"] = visual_logits
        outputs["pred_logits"] = refined_logits
        outputs["lqrf_residual"] = residual
        outputs["lqrf_has_records"] = (
            has_records
            if has_records is not None
            else torch.zeros(
                visual_logits.shape[0], dtype=torch.bool, device=visual_logits.device
            )
        )
        outputs["lqrf_gate"] = gate.reshape(1)
        if self.expose_attention and attention is not None:
            outputs["lqrf_attention"] = attention
        return outputs
