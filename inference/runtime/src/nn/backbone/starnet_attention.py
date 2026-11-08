"""StarNet backbone with lightweight attention on selected output features."""

from typing import Sequence

import torch
import torch.nn as nn

from ...core import register


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32, min_channels: int = 8) -> None:
        super().__init__()
        hidden = max(min_channels, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class EMAAttention(nn.Module):
    def __init__(self, channels: int, factor: int = 8) -> None:
        super().__init__()
        groups = min(factor, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        group_channels = channels // groups

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)

        x_h = group_x.mean(dim=3, keepdim=True)
        x_w = group_x.mean(dim=2, keepdim=True).transpose(2, 3)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.transpose(2, 3).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, 1, -1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, 1, -1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


@register()
class StarNetAttention(nn.Module):
    def __init__(
        self,
        name: str,
        return_layers: Sequence[str],
        pretrained: bool = False,
        exportable: bool = True,
        features_only: bool = True,
        attention_type: str = 'ca',
        attention_layers: Sequence[int] = (2, 3),
        ca_reduction: int = 32,
        ema_factor: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        assert features_only, 'StarNetAttention expects timm features_only=True'
        assert attention_type in ['ca', 'ema']

        import timm

        model = timm.create_model(
            name,
            pretrained=pretrained,
            exportable=exportable,
            features_only=features_only,
            **kwargs,
        )
        assert set(return_layers).issubset(model.feature_info.module_name()), \
            f'return_layers should be a subset of {model.feature_info.module_name()}'

        return_idx = [
            model.feature_info.module_name().index(name)
            for name in return_layers
        ]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers
        self.model = model

        attention_layer_set = set(attention_layers)
        modules = []
        for i, channels in enumerate(self.channels):
            if i not in attention_layer_set:
                modules.append(nn.Identity())
            elif attention_type == 'ca':
                modules.append(CoordAttention(channels, reduction=ca_reduction))
            else:
                modules.append(EMAAttention(channels, factor=ema_factor))
        self.attention = nn.ModuleList(modules)
        self.attention_type = attention_type
        self.attention_layers = list(attention_layers)

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        feats = []
        for out_pos, idx in enumerate(self.return_idx):
            feat = outputs[idx]
            expected_channels = self.channels[out_pos]
            if feat.ndim == 4 and feat.shape[1] != expected_channels and feat.shape[-1] == expected_channels:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(self.attention[out_pos](feat))
        return feats
