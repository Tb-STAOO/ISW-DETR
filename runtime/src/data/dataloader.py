"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    images = torch.cat([x[0][None] for x in items], dim=0)
    targets = [x[1] for x in items]
    modal_inputs = _collate_modal_inputs(targets)
    visual_inputs = _collate_visual_inputs(targets)
    if visual_inputs is not None:
        return images, targets, modal_inputs, visual_inputs
    if modal_inputs is None:
        return images, targets
    return images, targets, modal_inputs


def _collate_modal_inputs(targets):
    if not targets:
        return None

    modal_targets = [target for target in targets if 'weight_tokens' in target]
    if not modal_targets:
        return None

    source = modal_targets[0]
    token_dim = int(source['weight_tokens'].shape[-1])
    raw_flags = ['weight_tokens_raw' in target for target in modal_targets]
    if any(raw_flags) and not all(raw_flags):
        raise ValueError('weight_tokens_raw must be present for every sample that has weight_tokens.')
    has_raw_tokens = all(raw_flags)
    raw_dim = int(source['weight_tokens_raw'].shape[-1]) if has_raw_tokens else 0
    order_flags = ['event_order' in target for target in modal_targets]
    if any(order_flags) and not all(order_flags):
        raise ValueError('event_order must be present for every sample that has weight_tokens.')
    has_event_order = all(order_flags)

    for target in modal_targets:
        tokens = target['weight_tokens']
        if tokens.ndim != 2 or int(tokens.shape[-1]) != token_dim:
            raise ValueError('All weight_tokens tensors in a batch must have the same feature dimension.')
        if has_raw_tokens:
            raw_tokens = target['weight_tokens_raw']
            if (
                raw_tokens.ndim != 2
                or int(raw_tokens.shape[0]) != int(tokens.shape[0])
                or int(raw_tokens.shape[-1]) != raw_dim
            ):
                raise ValueError('weight_tokens_raw must match weight_tokens event count and feature dimension.')
        mask = target.get('weight_mask')
        if mask is not None and (mask.ndim != 1 or int(mask.shape[0]) != int(tokens.shape[0])):
            raise ValueError('weight_mask must match the weight token event count.')
        if has_event_order:
            order = target['event_order']
            if order.ndim != 1 or int(order.shape[0]) != int(tokens.shape[0]):
                raise ValueError('event_order must match the weight token event count.')

    max_events = max(int(target['weight_tokens'].shape[0]) for target in modal_targets)
    batch_size = len(targets)
    weight_tokens = source['weight_tokens'].new_zeros((batch_size, max_events, token_dim))
    weight_mask = torch.zeros((batch_size, max_events), dtype=torch.bool)
    weight_tokens_raw = (
        source['weight_tokens_raw'].new_zeros((batch_size, max_events, raw_dim))
        if has_raw_tokens
        else None
    )
    event_order = torch.zeros((batch_size, max_events), dtype=torch.int64) if has_event_order else None
    scene_total_weight_g = torch.full((batch_size,), torch.nan, dtype=torch.float32)
    empty_baseline_g = torch.full((batch_size,), torch.nan, dtype=torch.float32)
    scene_total_available = torch.zeros((batch_size,), dtype=torch.bool)
    scene_total_independent = torch.zeros((batch_size,), dtype=torch.bool)

    for i, target in enumerate(targets):
        if 'weight_tokens' not in target:
            if 'weight_tokens_raw' in target or 'weight_mask' in target:
                raise ValueError('weight_tokens_raw/weight_mask cannot be provided without weight_tokens.')
            continue
        tokens = target.pop('weight_tokens')
        raw_tokens = target.pop('weight_tokens_raw', None)
        mask = target.pop('weight_mask', None)
        order = target.pop('event_order', None)
        scene_total_weight_g[i] = target.pop(
            'scene_total_weight_g', scene_total_weight_g[i]
        )
        empty_baseline_g[i] = target.pop('empty_baseline_g', empty_baseline_g[i])
        scene_total_available[i] = target.pop(
            'scene_total_available', scene_total_available[i]
        )
        scene_total_independent[i] = target.pop(
            'scene_total_independent', scene_total_independent[i]
        )
        n = int(tokens.shape[0])
        if n == 0:
            continue
        weight_tokens[i, :n] = tokens
        if has_raw_tokens:
            weight_tokens_raw[i, :n] = raw_tokens
        if has_event_order:
            event_order[i, :n] = order
        if mask is None:
            weight_mask[i, :n] = True
        else:
            weight_mask[i, :n] = mask.to(dtype=torch.bool)

    modal_inputs = {
        'weight_tokens': weight_tokens,
        'weight_mask': weight_mask,
    }
    if has_raw_tokens:
        modal_inputs['weight_tokens_raw'] = weight_tokens_raw
    if has_event_order:
        modal_inputs['event_order'] = event_order
    modal_inputs.update({
        'scene_total_weight_g': scene_total_weight_g,
        'empty_baseline_g': empty_baseline_g,
        'scene_total_available': scene_total_available,
        'scene_total_independent': scene_total_independent,
    })
    return modal_inputs


def _collate_visual_inputs(targets):
    if not targets or not any('car_detail_image' in target for target in targets):
        return None

    first = next(target['car_detail_image'] for target in targets if 'car_detail_image' in target)
    channels = int(first.shape[0])
    sizes = []
    for target in targets:
        if 'car_detail_image' in target:
            size = target.get('car_detail_size')
            if size is None:
                size = torch.tensor(target['car_detail_image'].shape[-2:], dtype=torch.int64)
            sizes.append(size.to(dtype=torch.int64))
        else:
            sizes.append(torch.zeros(2, dtype=torch.int64))
    detail_sizes = torch.stack(sizes)
    max_height = int(detail_sizes[:, 0].max().item())
    max_width = int(detail_sizes[:, 1].max().item())
    detail_images = first.new_zeros((len(targets), channels, max_height, max_width))

    for index, target in enumerate(targets):
        detail = target.pop('car_detail_image', None)
        target.pop('car_detail_size', None)
        if detail is None:
            continue
        height, width = detail.shape[-2:]
        detail_images[index, :, :height, :width] = detail

    return {
        'detail_images': detail_images,
        'detail_sizes': detail_sizes,
    }


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        modal_inputs = _collate_modal_inputs(targets)
        visual_inputs = _collate_visual_inputs(targets)

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        if visual_inputs is not None:
            return images, targets, modal_inputs, visual_inputs
        if modal_inputs is None:
            return images, targets
        return images, targets, modal_inputs

