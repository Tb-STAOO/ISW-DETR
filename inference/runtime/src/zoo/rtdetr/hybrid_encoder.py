"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register


__all__ = [
    'HybridEncoder',
    'HybridEncoderP2DGF',
    'HybridEncoderP2DGFDSCA',
    'HybridEncoderP2DGFEnhanced',
    'HybridEncoderP2Guide',
]



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class P2DetailGuidedFusion(nn.Module):
    """Detail-guided fusion for the P3-to-P2 concat tensor."""

    def __init__(self, channels: int, reduction: int = 16, init_gamma: float = 1e-2) -> None:
        super().__init__()
        total_channels = int(channels)
        if total_channels % 2 != 0:
            raise ValueError('P2DetailGuidedFusion expects an even channel concat tensor.')

        c = total_channels // 2
        hidden = int(max(16, c // max(1, int(reduction))))
        self.branch_channels = c

        self.detail_refine = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.semantic_refine = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.spatial_gate = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2, bias=True)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(3 * c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic, lateral = torch.split(x, self.branch_channels, dim=1)

        semantic_smooth = F.avg_pool2d(semantic, kernel_size=3, stride=1, padding=1)
        lateral_smooth = F.avg_pool2d(lateral, kernel_size=3, stride=1, padding=1)
        semantic_detail = semantic - semantic_smooth
        lateral_detail = lateral - lateral_smooth
        mismatch = lateral - semantic

        detail_evidence = torch.abs(lateral_detail)
        mismatch_evidence = torch.abs(mismatch)
        semantic_evidence = torch.abs(semantic_detail)

        spatial_gate = torch.sigmoid(
            self.spatial_gate(
                torch.cat(
                    (
                        torch.mean(detail_evidence, dim=1, keepdim=True),
                        torch.mean(mismatch_evidence, dim=1, keepdim=True),
                        torch.mean(semantic_evidence, dim=1, keepdim=True),
                    ),
                    dim=1,
                )
            )
        )
        channel_gate = torch.sigmoid(
            self.channel_gate(
                torch.cat(
                    (
                        F.adaptive_avg_pool2d(detail_evidence, 1),
                        F.adaptive_avg_pool2d(mismatch_evidence, 1),
                        F.adaptive_avg_pool2d(semantic_evidence, 1),
                    ),
                    dim=1,
                )
            )
        )
        gate = spatial_gate * channel_gate

        detail_update = self.detail_refine(lateral_detail)
        semantic_update = self.semantic_refine(mismatch)
        semantic = semantic + self.gamma * gate * detail_update
        lateral = lateral + self.gamma * gate * semantic_update
        return torch.cat((semantic, lateral), dim=1)


class P2DySample(nn.Module):
    """Lightweight dynamic upsampling used only for the P3-to-P2 path."""

    def __init__(self, channels: int, scale: int = 2, groups: int = 4) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError('P2DySample currently supports scale=2 only.')
        if channels % groups != 0:
            raise ValueError('P2DySample channels must be divisible by groups.')
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset = nn.Conv2d(channels, 2 * groups * scale * scale, kernel_size=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer('init_pos', self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        half = (self.scale - 1) / 2
        pos = torch.arange(-half, half + 1, dtype=torch.float32) / self.scale
        yy, xx = torch.meshgrid(pos, pos, indexing='ij')
        pos = torch.stack((xx, yy), dim=0)
        pos = pos.reshape(1, 2, self.scale * self.scale, 1, 1)
        pos = pos.repeat(1, 1, self.groups, 1, 1)
        return pos.reshape(1, 2 * self.groups * self.scale * self.scale, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x) * 0.25 + self.init_pos.to(dtype=x.dtype, device=x.device)
        return self._sample(x, offset)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_y, coords_x = torch.meshgrid(
            torch.arange(h, dtype=x.dtype, device=x.device) + 0.5,
            torch.arange(w, dtype=x.dtype, device=x.device) + 0.5,
            indexing='ij')
        coords = torch.stack((coords_x, coords_y), dim=0).view(1, 2, 1, h, w)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, self.groups, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).reshape(b * self.groups, self.scale * h, self.scale * w, 2)
        x = x.reshape(b * self.groups, -1, h, w)
        x = F.grid_sample(x, coords, mode='bilinear', align_corners=False, padding_mode='border')
        return x.reshape(b, -1, self.scale * h, self.scale * w)


class P2TextureGate(nn.Module):
    """High-frequency texture enhancement for the P2 lateral branch."""

    def __init__(self, channels: int, reduction: int = 16, init_gamma: float = 1e-2) -> None:
        super().__init__()
        total_channels = int(channels)
        if total_channels % 2 != 0:
            raise ValueError('P2TextureGate expects an even channel concat tensor.')
        c = total_channels // 2
        hidden = int(max(16, c // max(1, int(reduction))))
        self.branch_channels = c
        self.texture_refine = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic, lateral = torch.split(x, self.branch_channels, dim=1)
        texture = lateral - F.avg_pool2d(lateral, kernel_size=3, stride=1, padding=1)
        texture_abs = torch.abs(texture)
        spatial_gate = torch.sigmoid(
            self.spatial_gate(torch.cat((
                torch.mean(texture_abs, dim=1, keepdim=True),
                torch.amax(texture_abs, dim=1, keepdim=True),
            ), dim=1))
        )
        channel_gate = torch.sigmoid(self.channel_gate(F.adaptive_avg_pool2d(texture_abs, 1)))
        lateral = lateral + self.gamma * spatial_gate * channel_gate * self.texture_refine(texture)
        return torch.cat((semantic, lateral), dim=1)


class P2SmallObjectFeatureGuidance(nn.Module):
    """Small-object response guidance for the P2 lateral branch.

    The concat tensor is expected to be [semantic, lateral].  The module learns
    a spatial response from P2 high-frequency evidence and uses the semantic
    branch as a weak location hint, then applies a very small residual update to
    the P2 branch.  Zero initialization keeps the initial behavior close to the
    original P2-DGF path.
    """

    def __init__(self, channels: int, reduction: int = 16, init_gamma: float = 1e-2) -> None:
        super().__init__()
        total_channels = int(channels)
        if total_channels % 2 != 0:
            raise ValueError('P2SmallObjectFeatureGuidance expects an even channel concat tensor.')
        c = total_channels // 2
        hidden = int(max(16, c // max(1, int(reduction))))
        self.branch_channels = c

        self.edge_refine = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.p2_response = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.semantic_response = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True),
        )

        nn.init.zeros_(self.edge_refine.weight)
        nn.init.zeros_(self.p2_response.weight)
        nn.init.zeros_(self.p2_response.bias)
        nn.init.zeros_(self.semantic_response.weight)
        nn.init.zeros_(self.semantic_response.bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic, lateral = torch.split(x, self.branch_channels, dim=1)

        lateral_detail = lateral - F.avg_pool2d(lateral, kernel_size=3, stride=1, padding=1)
        edge_evidence = torch.abs(lateral_detail)
        lateral_evidence = torch.abs(lateral)
        semantic_evidence = torch.abs(semantic)

        p2_response = torch.sigmoid(self.p2_response(torch.cat((
            torch.mean(edge_evidence, dim=1, keepdim=True),
            torch.amax(edge_evidence, dim=1, keepdim=True),
            torch.mean(lateral_evidence, dim=1, keepdim=True),
            torch.amax(lateral_evidence, dim=1, keepdim=True),
        ), dim=1)))
        semantic_response = torch.sigmoid(self.semantic_response(torch.cat((
            torch.mean(semantic_evidence, dim=1, keepdim=True),
            torch.amax(semantic_evidence, dim=1, keepdim=True),
        ), dim=1)))
        response = 0.5 * (p2_response + semantic_response)
        channel_gate = torch.sigmoid(self.channel_gate(F.adaptive_avg_pool2d(edge_evidence, 1)))

        residual = self.edge_refine(lateral_detail)
        lateral = lateral * (1.0 + self.gamma * (response - 0.5))
        lateral = lateral + self.gamma * response * channel_gate * residual
        return torch.cat((semantic, lateral), dim=1)


class P2ODConvLite(nn.Module):
    """P2-only dynamic depthwise convolution with a tiny expert pool."""

    def __init__(self, channels: int, kernel_num: int = 2, reduction: int = 16, init_gamma: float = 5e-3) -> None:
        super().__init__()
        if kernel_num < 1:
            raise ValueError('kernel_num must be >= 1.')
        hidden = int(max(16, channels // max(1, int(reduction))))
        self.kernel_num = int(kernel_num)
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
            for _ in range(self.kernel_num)
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.kernel_num, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.attn[-1].weight)
        nn.init.zeros_(self.attn[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x).flatten(1), dim=1)
        update = 0
        for i, expert in enumerate(self.experts):
            update = update + expert(x) * weights[:, i].view(-1, 1, 1, 1)
        return x + self.gamma * update


class P2DetailSemanticCrossAttention(nn.Module):
    def __init__(self, total_channels, num_heads=4, pool_size=20, init_gamma=1e-2):
        super().__init__()
        if total_channels % 2 != 0:
            raise ValueError('P2DetailSemanticCrossAttention expects an even channel concat tensor.')

        c = total_channels // 2
        if c % num_heads != 0:
            raise ValueError('branch channels must be divisible by num_heads.')

        self.branch_channels = c
        self.num_heads = num_heads
        self.pool_size = int(pool_size)
        self.head_dim = c // num_heads

        self.q_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.detail_refine = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

        nn.init.zeros_(self.out_proj.weight)

    def _flatten_heads(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic, lateral = torch.split(x, self.branch_channels, dim=1)
        lateral_detail = lateral - F.avg_pool2d(lateral, kernel_size=3, stride=1, padding=1)

        q = F.adaptive_avg_pool2d(self.detail_refine(lateral_detail), self.pool_size)
        k = F.adaptive_avg_pool2d(semantic, self.pool_size)
        v = F.adaptive_avg_pool2d(semantic, self.pool_size)

        q = self._flatten_heads(self.q_proj(q))
        k = self._flatten_heads(self.k_proj(k))
        v = self._flatten_heads(self.v_proj(v))

        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)

        b = x.shape[0]
        context = context.transpose(2, 3).reshape(
            b, self.branch_channels, self.pool_size, self.pool_size)
        context = F.interpolate(context, size=lateral.shape[-2:], mode='bilinear', align_corners=False)
        lateral = lateral + self.gamma * self.out_proj(context)

        return torch.cat((semantic, lateral), dim=1)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


@register()
class HybridEncoderP2DGF(HybridEncoder):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v1',
                 dgf_reduction=16,
                 dgf_init_gamma=1e-2):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )
        if len(in_channels) < 4 or feat_strides[0] != 4:
            raise ValueError('HybridEncoderP2DGF expects a P2/P3/P4/P5 feature pyramid.')
        self.p2_dgf = P2DetailGuidedFusion(hidden_dim * 2, dgf_reduction, dgf_init_gamma)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            concat_feat = torch.cat([upsample_feat, feat_low], dim=1)
            if idx == 1:
                concat_feat = self.p2_dgf(concat_feat)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](concat_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


@register()
class HybridEncoderP2DGFEnhanced(HybridEncoderP2DGF):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v1',
                 dgf_reduction=16,
                 dgf_init_gamma=1e-2,
                 p2_upsample='nearest',
                 p2_dysample_groups=4,
                 p2_use_texture_gate=False,
                 p2_texture_reduction=16,
                 p2_texture_init_gamma=1e-2,
                 p2_use_sofg=False,
                 p2_sofg_reduction=16,
                 p2_sofg_init_gamma=1e-2,
                 p2_use_odconv_lite=False,
                 p2_odconv_kernel_num=2,
                 p2_odconv_reduction=16,
                 p2_odconv_init_gamma=5e-3):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
            dgf_reduction=dgf_reduction,
            dgf_init_gamma=dgf_init_gamma,
        )
        if p2_upsample not in ('nearest', 'dysample'):
            raise ValueError("p2_upsample must be 'nearest' or 'dysample'.")
        self.p2_upsample = p2_upsample
        self.p2_upsampler = P2DySample(hidden_dim, scale=2, groups=p2_dysample_groups) \
            if p2_upsample == 'dysample' else None
        self.p2_texture_gate = P2TextureGate(
            hidden_dim * 2, p2_texture_reduction, p2_texture_init_gamma) \
            if p2_use_texture_gate else None
        self.p2_sofg = P2SmallObjectFeatureGuidance(
            hidden_dim * 2, p2_sofg_reduction, p2_sofg_init_gamma) \
            if p2_use_sofg else None
        self.p2_odconv_lite = P2ODConvLite(
            hidden_dim * 2, p2_odconv_kernel_num, p2_odconv_reduction, p2_odconv_init_gamma) \
            if p2_use_odconv_lite else None

    def _upsample_to_low(self, feat_heigh: torch.Tensor, feat_low: torch.Tensor, is_p2: bool) -> torch.Tensor:
        if is_p2 and self.p2_upsampler is not None:
            upsample_feat = self.p2_upsampler(feat_heigh)
            if upsample_feat.shape[-2:] != feat_low.shape[-2:]:
                upsample_feat = F.interpolate(upsample_feat, size=feat_low.shape[-2:], mode='bilinear', align_corners=False)
            return upsample_feat
        return F.interpolate(feat_heigh, size=feat_low.shape[-2:], mode='nearest')

    def _enhance_p2_concat(self, concat_feat: torch.Tensor) -> torch.Tensor:
        if self.p2_sofg is not None:
            concat_feat = self.p2_sofg(concat_feat)
        if self.p2_texture_gate is not None:
            concat_feat = self.p2_texture_gate(concat_feat)
        concat_feat = self.p2_dgf(concat_feat)
        if self.p2_odconv_lite is not None:
            concat_feat = self.p2_odconv_lite(concat_feat)
        return concat_feat

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            is_p2 = idx == 1
            upsample_feat = self._upsample_to_low(feat_heigh, feat_low, is_p2)
            concat_feat = torch.cat([upsample_feat, feat_low], dim=1)
            if is_p2:
                concat_feat = self._enhance_p2_concat(concat_feat)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](concat_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


@register()
class HybridEncoderP2DGFDSCA(HybridEncoderP2DGF):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v1',
                 dgf_reduction=16,
                 dgf_init_gamma=1e-2,
                 dsca_heads=4,
                 dsca_pool_size=20,
                 dsca_init_gamma=1e-2):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
            dgf_reduction=dgf_reduction,
            dgf_init_gamma=dgf_init_gamma,
        )
        self.p2_dsca = P2DetailSemanticCrossAttention(
            hidden_dim * 2,
            num_heads=dsca_heads,
            pool_size=dsca_pool_size,
            init_gamma=dsca_init_gamma,
        )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            concat_feat = torch.cat([upsample_feat, feat_low], dim=1)
            if idx == 1:
                concat_feat = self.p2_dsca(concat_feat)
                concat_feat = self.p2_dgf(concat_feat)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](concat_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


@register()
class HybridEncoderP2Guide(HybridEncoder):
    """P3-P5 encoder with training-only P2 detail-energy distillation."""

    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[24, 48, 96, 192],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v1',
                 p2_distill_weight=0.25):
        if len(in_channels) != 4 or len(feat_strides) != 4 or feat_strides[0] != 4:
            raise ValueError('HybridEncoderP2Guide expects P2/P3/P4/P5 backbone features.')
        self.backbone_in_channels = list(in_channels)
        self.backbone_feat_strides = list(feat_strides)
        self.p2_distill_weight = float(p2_distill_weight)
        super().__init__(
            in_channels=in_channels[1:],
            feat_strides=feat_strides[1:],
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )

    @staticmethod
    def _detail_energy(feat: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        return torch.abs(feat - smooth).mean(dim=1, keepdim=True)

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('HybridEncoderP2Guide expects four backbone feature maps.')
        p2 = feats[0]
        outs = super().forward(feats[1:])
        if not self.training:
            return outs

        teacher = self._detail_energy(p2).detach()
        teacher = F.interpolate(
            teacher,
            size=outs[0].shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        student = self._detail_energy(outs[0])
        teacher = F.normalize(teacher.flatten(1), dim=1)
        student = F.normalize(student.flatten(1), dim=1)
        loss = (1.0 - (teacher * student).sum(dim=1)).mean()
        return outs, loss * self.p2_distill_weight
