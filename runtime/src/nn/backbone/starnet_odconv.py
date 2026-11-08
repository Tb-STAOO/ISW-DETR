"""StarNet backbone with selective ODConv injection.

This module keeps the original timm StarNet topology and replaces only the
selected depthwise convolutions inside later StarBlocks. It is intended for
controlled ablation against the B3 StarNet-S1 + P2-DGF baseline.
"""

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


class ODConv2d(nn.Module):
    """Omni-dimensional dynamic convolution.

    The implementation supports grouped/depthwise convolution and is used here
    as a lightweight dynamic replacement for StarNet's 7x7 depthwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        kernel_num: int = 4,
        reduction: float = 0.0625,
        min_channel: int = 16,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        hidden_channels = max(int(in_channels * reduction), min_channel)
        self.weight = nn.Parameter(
            torch.empty(
                kernel_num,
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
            )
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.channel_fc = nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        self.filter_fc = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)
        self.spatial_fc = nn.Conv2d(
            hidden_channels,
            kernel_size[0] * kernel_size[1],
            1,
            bias=True,
        )
        self.kernel_fc = nn.Conv2d(hidden_channels, kernel_num, 1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        for layer in [self.channel_fc, self.filter_fc, self.spatial_fc]:
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, 2.0)
        nn.init.zeros_(self.kernel_fc.weight)
        nn.init.zeros_(self.kernel_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = self.reduce(self.avg_pool(x))

        channel_attn = torch.sigmoid(self.channel_fc(attn))
        filter_attn = torch.sigmoid(self.filter_fc(attn))
        spatial_attn = torch.sigmoid(self.spatial_fc(attn)).view(
            b,
            1,
            1,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        kernel_attn = torch.softmax(self.kernel_fc(attn), dim=1).view(
            b,
            self.kernel_num,
            1,
            1,
            1,
            1,
        )

        x = x * channel_attn
        aggregate_weight = (
            kernel_attn * spatial_attn * self.weight.unsqueeze(0)
        ).sum(dim=1)
        aggregate_weight = aggregate_weight.view(
            b * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        x = x.reshape(1, b * c, h, w)
        out = F.conv2d(
            x,
            aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        out = out.view(b, self.out_channels, out.shape[-2], out.shape[-1])
        return out * filter_attn


class ODDepthwiseConvBN(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
        dilation: int = 1,
        kernel_num: int = 4,
        reduction: float = 0.0625,
        min_channel: int = 16,
        with_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = ODConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            kernel_num=kernel_num,
            reduction=reduction,
            min_channel=min_channel,
        )
        self.bn = nn.BatchNorm2d(channels) if with_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


@register()
class StarNetODConv(nn.Module):
    def __init__(
        self,
        name: str,
        return_layers: Sequence[str],
        pretrained: bool = False,
        exportable: bool = True,
        features_only: bool = True,
        od_stages: Iterable[int] = (2, 3),
        od_target: str = 'dwconv2',
        od_kernel_num: int = 4,
        od_reduction: float = 0.0625,
        od_min_channel: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        assert features_only, 'StarNetODConv expects timm features_only=True'
        assert od_target in ['dwconv', 'dwconv2', 'both']

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

        self.num_odconv_blocks = self._inject_odconv(
            od_stages=od_stages,
            od_target=od_target,
            od_kernel_num=od_kernel_num,
            od_reduction=od_reduction,
            od_min_channel=od_min_channel,
        )
        assert self.num_odconv_blocks > 0, 'No StarNet blocks were replaced by ODConv'

    def _replace_depthwise_conv(
        self,
        module: nn.Module,
        kernel_num: int,
        reduction: float,
        min_channel: int,
    ) -> ODDepthwiseConvBN:
        conv = module.conv
        with_bn = hasattr(module, 'bn') and module.bn is not None
        return ODDepthwiseConvBN(
            channels=conv.in_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            kernel_num=kernel_num,
            reduction=reduction,
            min_channel=min_channel,
            with_bn=with_bn,
        )

    def _inject_odconv(
        self,
        od_stages: Iterable[int],
        od_target: str,
        od_kernel_num: int,
        od_reduction: float,
        od_min_channel: int,
    ) -> int:
        replaced = 0
        for stage_idx in od_stages:
            stage = getattr(self.model, f'stages_{stage_idx}', None)
            if stage is None:
                continue
            for block in stage:
                if od_target in ['dwconv', 'both'] and hasattr(block, 'dwconv'):
                    block.dwconv = self._replace_depthwise_conv(
                        block.dwconv,
                        od_kernel_num,
                        od_reduction,
                        od_min_channel,
                    )
                    replaced += 1
                if od_target in ['dwconv2', 'both'] and hasattr(block, 'dwconv2'):
                    block.dwconv2 = self._replace_depthwise_conv(
                        block.dwconv2,
                        od_kernel_num,
                        od_reduction,
                        od_min_channel,
                    )
                    replaced += 1
        return replaced

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        feats = []
        for idx in self.return_idx:
            feat = outputs[idx]
            expected_channels = self.channels[self.return_idx.index(idx)]
            if feat.ndim == 4 and feat.shape[1] != expected_channels and feat.shape[-1] == expected_channels:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(feat)
        return feats
