"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
import json
from collections import OrderedDict
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from ...core import register

__all__ = [
    'RTDETRTransformerv2',
    'RTDETRTransformerv2SQC',
    'ScaleWeightTokenEncoder',
    'MMRTDETRTransformerv2',
    'EventQueryMMRTDETRTransformerv2',
    'WeightScenePriorMMRTDETRTransformerv2',
    'QueryAlignedWeightPriorMMRTDETRTransformerv2',
]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ScaleQueryCalibrator(nn.Module):
    def __init__(self, hidden_dim=256, hidden_ratio=0.25, init_gamma=1e-2):
        super().__init__()
        hidden = max(16, int(hidden_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, content: torch.Tensor, ref_points_unact: torch.Tensor) -> torch.Tensor:
        ref_boxes = torch.sigmoid(ref_points_unact.detach()).clamp(min=1e-4, max=1.0)
        wh = ref_boxes[..., 2:].clamp(min=1e-4)
        w, h = wh.unbind(-1)
        area = (w * h).sqrt().clamp(min=1e-4)
        aspect = (w / h).clamp(min=1e-4, max=1e4)
        scale_feat = torch.stack(
            (torch.log(w), torch.log(h), torch.log(area), torch.log(aspect)),
            dim=-1,
        )

        shift, gate = self.mlp(scale_feat).chunk(2, dim=-1)
        return content + self.gamma * torch.sigmoid(gate) * shift


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_features = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)
            dec_out_features.append(output)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_features),
        )


@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True, 
                 cross_attn_method='default', 
                 query_select_method='default',
                 confusion_map_path=None,
                 confusion_dn_probability=0.0,
                 expose_query_features=False):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.confusion_dn_probability = float(confusion_dn_probability)
        self.expose_query_features = bool(expose_query_features)

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.register_buffer(
            'confusion_candidates',
            self._load_confusion_candidates(confusion_map_path, num_classes),
            persistent=False,
        )

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()

    @staticmethod
    def _load_confusion_candidates(confusion_map_path, num_classes):
        if not confusion_map_path:
            return torch.full((num_classes, 1), -1, dtype=torch.long)

        path = Path(confusion_map_path)
        if not path.is_file():
            raise FileNotFoundError(f'Confusion map not found: {path}')
        with path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        mapping = raw.get('confusion_candidates', raw)
        max_candidates = max(
            [len(v) for v in mapping.values() if isinstance(v, list)] or [1]
        )
        candidates = torch.full(
            (num_classes, max_candidates),
            -1,
            dtype=torch.long,
        )
        for label, values in mapping.items():
            label = int(label)
            if label < 0 or label >= num_classes or not isinstance(values, list):
                continue
            clean = [
                int(v) for v in values
                if 0 <= int(v) < num_classes and int(v) != label
            ][:max_candidates]
            if clean:
                candidates[label, :len(clean)] = torch.tensor(clean, dtype=torch.long)
        return candidates
        
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:            
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords


    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.expose_query_features:
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


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]


class ScaleWeightTokenEncoder(nn.Module):
    """Encode ordered scale-event features into decoder hidden tokens."""

    def __init__(
        self,
        input_dim=8,
        hidden_dim=256,
        num_layers=2,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
        activation='gelu',
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, weight_tokens, weight_mask):
        x = self.input_proj(weight_tokens)
        key_padding_mask = ~weight_mask.to(dtype=torch.bool)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.output_norm(x)


class ScaleWeightMLPEncoder(nn.Module):
    """Encode each scale event independently for the no-attention ablation."""

    def __init__(self, input_dim=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, weight_tokens, weight_mask):
        del weight_mask
        return self.encoder(weight_tokens)

@register()
class MMRTDETRTransformerv2(RTDETRTransformerv2):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 confusion_map_path=None,
                 confusion_dn_probability=0.0,
                 expose_query_features=False,
                 weight_input_dim=8,
                 weight_encoder_type='transformer',
                 weight_encoder_layers=2,
                 weight_encoder_heads=4,
                 weight_encoder_ffn_dim=512,
                 weight_dropout=0.1,
                 modal_fusion_init_gamma=0.0,
                 modal_fusion_max_gamma=None,
                 event_align_temperature=1.0,
                 fuse_denoising_queries=False):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            confusion_map_path=confusion_map_path,
            confusion_dn_probability=confusion_dn_probability,
            expose_query_features=expose_query_features,
        )
        assert weight_encoder_type in ('transformer', 'mlp'), 'unsupported weight_encoder_type'
        self.weight_encoder_type = weight_encoder_type
        if weight_encoder_type == 'mlp':
            self.weight_token_encoder = ScaleWeightMLPEncoder(
                input_dim=weight_input_dim,
                hidden_dim=hidden_dim,
                dropout=weight_dropout,
            )
        else:
            self.weight_token_encoder = ScaleWeightTokenEncoder(
                input_dim=weight_input_dim,
                hidden_dim=hidden_dim,
                num_layers=weight_encoder_layers,
                nhead=weight_encoder_heads,
                dim_feedforward=weight_encoder_ffn_dim,
                dropout=weight_dropout,
            )
        self.modal_query_norm = nn.LayerNorm(hidden_dim)
        self.modal_key_norm = nn.LayerNorm(hidden_dim)
        self.modal_attn = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.modal_out_norm = nn.LayerNorm(hidden_dim)
        self.modal_gamma = nn.Parameter(torch.tensor(float(modal_fusion_init_gamma)))
        self.modal_fusion_max_gamma = None if modal_fusion_max_gamma is None else float(modal_fusion_max_gamma)
        self.event_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.event_token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.event_align_temperature = max(float(event_align_temperature), 1e-6)
        self.fuse_denoising_queries = bool(fuse_denoising_queries)

    def _encode_modal_inputs(self, modal_inputs, dtype, device):
        if modal_inputs is None:
            return None, None

        weight_tokens = modal_inputs.get('weight_tokens')
        weight_mask = modal_inputs.get('weight_mask')
        if weight_tokens is None or weight_mask is None:
            return None, None

        weight_tokens = weight_tokens.to(device=device, dtype=dtype)
        weight_mask = weight_mask.to(device=device, dtype=torch.bool)
        if weight_tokens.shape[1] == 0:
            weight_tokens = weight_tokens.new_zeros((weight_tokens.shape[0], 1, weight_tokens.shape[-1]))
            weight_mask = torch.ones((weight_tokens.shape[0], 1), dtype=torch.bool, device=device)

        empty_rows = ~weight_mask.any(dim=1)
        if empty_rows.any():
            weight_mask = weight_mask.clone()
            weight_mask[empty_rows, 0] = True

        return self.weight_token_encoder(weight_tokens, weight_mask), weight_mask

    def _fuse_modal_content(self, content, encoded_weight_tokens, weight_mask, denoising_bbox_unact=None):
        if encoded_weight_tokens is None or weight_mask is None:
            return content

        dn_count = 0
        if denoising_bbox_unact is not None and not self.fuse_denoising_queries:
            dn_count = int(denoising_bbox_unact.shape[1])

        prefix = content[:, :dn_count] if dn_count > 0 else None
        query = content[:, dn_count:] if dn_count > 0 else content
        modal_update, _ = self.modal_attn(
            self.modal_query_norm(query),
            self.modal_key_norm(encoded_weight_tokens),
            encoded_weight_tokens,
            key_padding_mask=~weight_mask,
            need_weights=False,
        )
        modal_gamma = self.modal_gamma
        if self.modal_fusion_max_gamma is not None:
            modal_gamma = modal_gamma.clamp(min=0.0, max=self.modal_fusion_max_gamma)
        query = query + modal_gamma * self.modal_out_norm(modal_update)
        if prefix is None:
            return query
        return torch.concat([prefix, query], dim=1)

    def _build_event_align_logits(self, query_features, encoded_weight_tokens, weight_mask):
        if encoded_weight_tokens is None or weight_mask is None:
            return None

        query = F.normalize(self.event_query_proj(query_features), dim=-1)
        tokens = F.normalize(self.event_token_proj(encoded_weight_tokens), dim=-1)
        logits = torch.matmul(query, tokens.transpose(1, 2)) / self.event_align_temperature
        logits = logits.masked_fill(~weight_mask[:, None, :], -1e4)
        return logits

    def forward(self, feats, targets=None, modal_inputs=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        encoded_weight_tokens, weight_mask = self._encode_modal_inputs(
            modal_inputs,
            dtype=memory.dtype,
            device=memory.device,
        )
        init_ref_contents = self._fuse_modal_content(
            init_ref_contents,
            encoded_weight_tokens,
            weight_mask,
            denoising_bbox_unact=denoising_bbox_unact,
        )

        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        event_align_logits = self._build_event_align_logits(out_features[-1], encoded_weight_tokens, weight_mask)
        if event_align_logits is not None:
            out['event_align_logits'] = event_align_logits

        if self.expose_query_features:
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


@register()
class EventQueryMMRTDETRTransformerv2(MMRTDETRTransformerv2):
    """Event-conditioned multimodal decoder for ScaleVision-MM.

    The first non-denoising decoder slots are softly conditioned by ordered
    weight events, while the remaining slots keep the original visual proposal
    behavior. This keeps COCO-style detection output unchanged.
    """

    def __init__(
        self,
        *args,
        event_query_slots=None,
        event_query_init_gamma=0.05,
        event_query_max_gamma=0.05,
        event_ref_init_gamma=0.05,
        event_ref_max_gamma=0.05,
        class_prior_init_alpha=0.05,
        class_prior_max_alpha=0.05,
        class_prior_logit_clip=4.0,
        reliability_gate_hidden=64,
        reliability_gate_init_bias=0.0,
        event_query_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hidden_dim
        weight_input_dim = int(kwargs.get('weight_input_dim', 8))

        self.event_query_slots = event_query_slots
        self.event_query_gamma = nn.Parameter(torch.tensor(float(event_query_init_gamma)))
        self.event_query_max_gamma = float(event_query_max_gamma)
        self.event_ref_gamma = nn.Parameter(torch.tensor(float(event_ref_init_gamma)))
        self.event_ref_max_gamma = float(event_ref_max_gamma)
        self.class_prior_alpha = nn.Parameter(torch.tensor(float(class_prior_init_alpha)))
        self.class_prior_max_alpha = float(class_prior_max_alpha)
        self.class_prior_logit_clip = float(class_prior_logit_clip)
        self.event_query_loss_weight = float(event_query_loss_weight)

        self.event_query_norm = nn.LayerNorm(hidden_dim)
        self.proposal_key_norm = nn.LayerNorm(hidden_dim)
        self.event_to_proposal_attn = nn.MultiheadAttention(
            hidden_dim,
            self.nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.event_query_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_query_out_norm = nn.LayerNorm(hidden_dim)

        self.weight_class_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

        gate_hidden = max(8, int(reliability_gate_hidden))
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(weight_input_dim),
            nn.Linear(weight_input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.constant_(self.reliability_gate[-1].bias, float(reliability_gate_init_bias))

    def _event_slot_count(self, encoded_weight_tokens, weight_mask, visual_query_count):
        if encoded_weight_tokens is None or weight_mask is None:
            return 0
        slots = min(int(encoded_weight_tokens.shape[1]), int(weight_mask.shape[1]), int(visual_query_count))
        if self.event_query_slots is not None:
            slots = min(slots, int(self.event_query_slots))
        return max(0, slots)

    def _inject_event_queries(
        self,
        content,
        ref_points_unact,
        encoded_weight_tokens,
        weight_mask,
        denoising_bbox_unact=None,
    ):
        if encoded_weight_tokens is None or weight_mask is None:
            return content, ref_points_unact, None

        dn_count = 0
        if denoising_bbox_unact is not None:
            dn_count = int(denoising_bbox_unact.shape[1])

        prefix_content = content[:, :dn_count] if dn_count > 0 else None
        prefix_ref = ref_points_unact[:, :dn_count] if dn_count > 0 else None
        visual_content = content[:, dn_count:] if dn_count > 0 else content
        visual_ref = ref_points_unact[:, dn_count:] if dn_count > 0 else ref_points_unact

        slots = self._event_slot_count(encoded_weight_tokens, weight_mask, visual_content.shape[1])
        if slots <= 0:
            return content, ref_points_unact, None

        event_tokens = encoded_weight_tokens[:, :slots]
        event_mask = weight_mask[:, :slots]
        attn_update, attn_weights = self.event_to_proposal_attn(
            self.event_query_norm(event_tokens),
            self.proposal_key_norm(visual_content),
            visual_content,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.event_query_fuser(torch.concat([event_tokens, attn_update], dim=-1))
        query_gamma = self.event_query_gamma.clamp(min=0.0, max=self.event_query_max_gamma)
        candidate_content = visual_content[:, :slots] + query_gamma * self.event_query_out_norm(fused)
        valid = event_mask.unsqueeze(-1).to(dtype=visual_content.dtype)
        event_content = torch.where(valid > 0, candidate_content, visual_content[:, :slots])
        if slots < visual_content.shape[1]:
            visual_content = torch.concat([event_content, visual_content[:, slots:]], dim=1)
        else:
            visual_content = event_content

        visual_ref_box = torch.sigmoid(visual_ref).clamp(min=1e-4, max=1.0 - 1e-4)
        event_ref_box = torch.matmul(attn_weights.to(visual_ref_box.dtype), visual_ref_box)
        ref_gamma = self.event_ref_gamma.clamp(min=0.0, max=self.event_ref_max_gamma)
        candidate_ref = visual_ref_box[:, :slots] * (1.0 - ref_gamma) + event_ref_box[:, :slots] * ref_gamma
        candidate_ref = candidate_ref.clamp(min=1e-4, max=1.0 - 1e-4)
        event_ref = torch.where(valid > 0, inverse_sigmoid(candidate_ref), visual_ref[:, :slots])
        if slots < visual_ref.shape[1]:
            visual_ref = torch.concat([event_ref, visual_ref[:, slots:]], dim=1)
        else:
            visual_ref = event_ref

        if prefix_content is not None:
            content = torch.concat([prefix_content, visual_content], dim=1)
            ref_points_unact = torch.concat([prefix_ref, visual_ref], dim=1)
        else:
            content = visual_content
            ref_points_unact = visual_ref

        return content, ref_points_unact, event_mask

    def _build_class_prior(self, encoded_weight_tokens, weight_mask, raw_weight_tokens, query_count):
        slots = self._event_slot_count(encoded_weight_tokens, weight_mask, query_count)
        if slots <= 0:
            return None, None

        prior = self.weight_class_head(encoded_weight_tokens[:, :slots])
        if self.class_prior_logit_clip > 0:
            prior = prior.clamp(min=-self.class_prior_logit_clip, max=self.class_prior_logit_clip)
        if raw_weight_tokens is not None:
            raw = raw_weight_tokens[:, :slots].to(device=prior.device, dtype=prior.dtype)
            reliability = torch.sigmoid(self.reliability_gate(raw))
        else:
            reliability = torch.ones((*prior.shape[:2], 1), dtype=prior.dtype, device=prior.device)
        reliability = reliability * weight_mask[:, :slots].unsqueeze(-1).to(dtype=prior.dtype)
        return prior * reliability, weight_mask[:, :slots]

    def _apply_class_prior(self, out_logits, encoded_weight_tokens, weight_mask, raw_weight_tokens):
        prior, event_mask = self._build_class_prior(
            encoded_weight_tokens,
            weight_mask,
            raw_weight_tokens,
            out_logits.shape[2],
        )
        if prior is None:
            return out_logits, None, None

        slots = prior.shape[1]
        alpha = self.class_prior_alpha.clamp(min=0.0, max=self.class_prior_max_alpha)
        logits_prefix = out_logits[:, :, :slots] + alpha * prior.unsqueeze(0)
        if slots < out_logits.shape[2]:
            out_logits = torch.concat([logits_prefix, out_logits[:, :, slots:]], dim=2)
        else:
            out_logits = logits_prefix
        return out_logits, event_mask, prior

    def forward(self, feats, targets=None, modal_inputs=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        encoded_weight_tokens, weight_mask = self._encode_modal_inputs(
            modal_inputs,
            dtype=memory.dtype,
            device=memory.device,
        )
        raw_weight_tokens = None
        if modal_inputs is not None and modal_inputs.get('weight_tokens') is not None:
            raw_weight_tokens = modal_inputs['weight_tokens'].to(device=memory.device, dtype=memory.dtype)

        init_ref_contents = self._fuse_modal_content(
            init_ref_contents,
            encoded_weight_tokens,
            weight_mask,
            denoising_bbox_unact=denoising_bbox_unact,
        )
        init_ref_contents, init_ref_points_unact, injected_event_mask = self._inject_event_queries(
            init_ref_contents,
            init_ref_points_unact,
            encoded_weight_tokens,
            weight_mask,
            denoising_bbox_unact=denoising_bbox_unact,
        )

        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        out_logits, event_mask, weight_event_logits = self._apply_class_prior(
            out_logits,
            encoded_weight_tokens,
            weight_mask,
            raw_weight_tokens,
        )
        if event_mask is None:
            event_mask = injected_event_mask

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        event_align_logits = self._build_event_align_logits(out_features[-1], encoded_weight_tokens, weight_mask)
        if event_align_logits is not None:
            out['event_align_logits'] = event_align_logits

        if event_mask is not None:
            slots = int(event_mask.shape[1])
            out['event_query_logits'] = out_logits[-1, :, :slots]
            out['event_query_boxes'] = out_bboxes[-1, :, :slots]
            out['event_query_mask'] = event_mask
        if weight_event_logits is not None:
            out['weight_event_logits'] = weight_event_logits

        if self.expose_query_features:
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


@register()
class WeightScenePriorMMRTDETRTransformerv2(MMRTDETRTransformerv2):
    """Use scale events as a scene-level class prior for RT-DETRv2 logits."""

    def __init__(
        self,
        *args,
        class_prior_init_alpha=0.02,
        class_prior_max_alpha=0.05,
        class_prior_logit_clip=4.0,
        prior_pooling='logsumexp',
        prior_temperature=1.0,
        use_reliability_gate=False,
        reliability_gate_hidden=64,
        reliability_gate_init_bias=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hidden_dim
        weight_input_dim = int(kwargs.get('weight_input_dim', 8))
        assert prior_pooling in ('max', 'mean', 'logsumexp'), 'unsupported prior_pooling'

        self.class_prior_alpha = nn.Parameter(torch.tensor(float(class_prior_init_alpha)))
        self.class_prior_max_alpha = float(class_prior_max_alpha)
        self.class_prior_logit_clip = float(class_prior_logit_clip)
        self.prior_pooling = prior_pooling
        self.prior_temperature = max(float(prior_temperature), 1e-6)
        self.use_reliability_gate = bool(use_reliability_gate)

        self.weight_class_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )
        if self.use_reliability_gate:
            gate_hidden = max(8, int(reliability_gate_hidden))
            self.reliability_gate = nn.Sequential(
                nn.LayerNorm(weight_input_dim),
                nn.Linear(weight_input_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.reliability_gate[-1].bias, float(reliability_gate_init_bias))
        else:
            self.reliability_gate = None

    def _build_weight_event_logits(self, encoded_weight_tokens, weight_mask, raw_weight_tokens):
        if encoded_weight_tokens is None or weight_mask is None:
            return None
        logits = self.weight_class_head(encoded_weight_tokens)
        mask = weight_mask.to(dtype=torch.bool)
        if self.reliability_gate is not None and raw_weight_tokens is not None:
            raw = raw_weight_tokens.to(device=logits.device, dtype=logits.dtype)
            reliability = torch.sigmoid(self.reliability_gate(raw))
            logits = logits * reliability
        return logits.masked_fill(~mask.unsqueeze(-1), -1e4)

    def _pool_scene_prior(self, event_logits, weight_mask):
        mask = weight_mask.to(dtype=torch.bool)
        if event_logits is None or not mask.any():
            return None
        if self.prior_pooling == 'max':
            return event_logits.max(dim=1).values
        if self.prior_pooling == 'mean':
            valid = mask.unsqueeze(-1).to(dtype=event_logits.dtype)
            clean = event_logits.masked_fill(~mask.unsqueeze(-1), 0.0)
            return clean.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        scaled = event_logits / self.prior_temperature
        prior = torch.logsumexp(scaled, dim=1) * self.prior_temperature
        empty = ~mask.any(dim=1)
        if empty.any():
            prior = prior.clone()
            prior[empty] = 0.0
        return prior

    def forward(self, feats, targets=None, modal_inputs=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        encoded_weight_tokens, weight_mask = self._encode_modal_inputs(
            modal_inputs,
            dtype=memory.dtype,
            device=memory.device,
        )
        raw_weight_tokens = None
        if modal_inputs is not None and modal_inputs.get('weight_tokens') is not None:
            raw_weight_tokens = modal_inputs['weight_tokens'].to(device=memory.device, dtype=memory.dtype)

        init_ref_contents = self._fuse_modal_content(
            init_ref_contents,
            encoded_weight_tokens,
            weight_mask,
            denoising_bbox_unact=denoising_bbox_unact,
        )

        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        weight_event_logits = self._build_weight_event_logits(
            encoded_weight_tokens,
            weight_mask,
            raw_weight_tokens,
        )
        scene_prior = self._pool_scene_prior(weight_event_logits, weight_mask) if weight_event_logits is not None else None
        if scene_prior is not None:
            if self.class_prior_logit_clip > 0:
                scene_prior = scene_prior.clamp(min=-self.class_prior_logit_clip, max=self.class_prior_logit_clip)
            alpha = self.class_prior_alpha.clamp(min=0.0, max=self.class_prior_max_alpha)
            out_logits = out_logits + alpha * scene_prior.unsqueeze(0).unsqueeze(2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        event_align_logits = self._build_event_align_logits(out_features[-1], encoded_weight_tokens, weight_mask)
        if event_align_logits is not None:
            out['event_align_logits'] = event_align_logits
        if weight_event_logits is not None:
            out['weight_event_logits'] = weight_event_logits

        if self.expose_query_features:
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


@register()
class RTDETRTransformerv2SQC(RTDETRTransformerv2):
    def __init__(self,
                 *args,
                 sqc_hidden_ratio=0.25,
                 sqc_init_gamma=1e-2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_query_calibrator = ScaleQueryCalibrator(
            hidden_dim=self.hidden_dim,
            hidden_ratio=sqc_hidden_ratio,
            init_gamma=sqc_init_gamma,
        )

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):
        content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list = super()._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )
        content = self.scale_query_calibrator(content, enc_topk_bbox_unact)
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list


@register()
class QueryAlignedWeightPriorMMRTDETRTransformerv2(WeightScenePriorMMRTDETRTransformerv2):
    """Apply event-specific weight class priors to visually matched queries.

    Unlike the scene-level prior, this decoder first predicts a query-to-weight
    event assignment and then injects the selected event class prior into each
    detection query. The assignment can be supervised with loss_event_align.
    """

    def __init__(
        self,
        *args,
        query_prior_temperature=0.5,
        detach_event_alignment=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.query_prior_temperature = max(float(query_prior_temperature), 1e-6)
        self.detach_event_alignment = bool(detach_event_alignment)

    def _build_query_prior(self, event_align_logits, weight_event_logits, weight_mask):
        if event_align_logits is None or weight_event_logits is None or weight_mask is None:
            return None

        logits = event_align_logits
        if self.detach_event_alignment:
            logits = logits.detach()
        attn = torch.softmax(logits / self.query_prior_temperature, dim=-1)
        valid = weight_mask.to(device=attn.device, dtype=attn.dtype).unsqueeze(1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        clean_event_logits = weight_event_logits.masked_fill(~weight_mask.to(torch.bool).unsqueeze(-1), 0.0)
        return torch.matmul(attn, clean_event_logits)

    def forward(self, feats, targets=None, modal_inputs=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    confusion_candidates=self.confusion_candidates,
                    confusion_probability=self.confusion_dn_probability)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        encoded_weight_tokens, weight_mask = self._encode_modal_inputs(
            modal_inputs,
            dtype=memory.dtype,
            device=memory.device,
        )
        raw_weight_tokens = None
        if modal_inputs is not None and modal_inputs.get('weight_tokens') is not None:
            raw_weight_tokens = modal_inputs['weight_tokens'].to(device=memory.device, dtype=memory.dtype)

        init_ref_contents = self._fuse_modal_content(
            init_ref_contents,
            encoded_weight_tokens,
            weight_mask,
            denoising_bbox_unact=denoising_bbox_unact,
        )

        out_bboxes, out_logits, out_features = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_features = torch.split(out_features, dn_meta['dn_num_split'], dim=2)

        weight_event_logits = self._build_weight_event_logits(
            encoded_weight_tokens,
            weight_mask,
            raw_weight_tokens,
        )
        event_align_logits = self._build_event_align_logits(out_features[-1], encoded_weight_tokens, weight_mask)
        query_prior = self._build_query_prior(event_align_logits, weight_event_logits, weight_mask)
        if query_prior is not None:
            if self.class_prior_logit_clip > 0:
                query_prior = query_prior.clamp(min=-self.class_prior_logit_clip, max=self.class_prior_logit_clip)
            alpha = self.class_prior_alpha.clamp(min=0.0, max=self.class_prior_max_alpha)
            out_logits = out_logits.clone()
            out_logits[-1] = out_logits[-1] + alpha * query_prior

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if event_align_logits is not None:
            out['event_align_logits'] = event_align_logits
        if weight_event_logits is not None:
            out['weight_event_logits'] = weight_event_logits

        if self.expose_query_features:
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