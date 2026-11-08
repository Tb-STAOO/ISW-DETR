"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _unpack_batch(batch):
    if len(batch) == 2:
        samples, targets = batch
        return samples, targets, None, None
    if len(batch) == 3:
        samples, targets, modal_inputs = batch
        return samples, targets, modal_inputs, None
    if len(batch) == 4:
        samples, targets, modal_inputs, visual_inputs = batch
        return samples, targets, modal_inputs, visual_inputs
    raise ValueError(f'Expected a 2-, 3-, or 4-item batch, got {len(batch)} items.')


def _move_targets_to_device(targets, device):
    return [
        {k: v.to(device) for k, v in t.items()}
        for t in targets
    ]


def _move_modal_inputs_to_device(modal_inputs, device):
    if modal_inputs is None:
        return None
    return {
        k: v.to(device) if hasattr(v, 'to') else v
        for k, v in modal_inputs.items()
    }


def _move_visual_inputs_to_device(visual_inputs, device):
    if visual_inputs is None:
        return None
    return {
        k: v.to(device) if hasattr(v, 'to') else v
        for k, v in visual_inputs.items()
    }


def _forward_model(model, samples, targets=None, modal_inputs=None, visual_inputs=None):
    if modal_inputs is None and visual_inputs is None:
        if targets is None:
            return model(samples)
        return model(samples, targets=targets)
    if targets is None and visual_inputs is None:
        return model(samples, modal_inputs=modal_inputs)
    if targets is None:
        return model(samples, modal_inputs=modal_inputs, visual_inputs=visual_inputs)
    return model(
        samples,
        targets=targets,
        modal_inputs=modal_inputs,
        visual_inputs=visual_inputs,
    )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    gradient_accumulation_steps = max(1, int(kwargs.get('gradient_accumulation_steps', 1)))
    num_batches = len(data_loader)
    updates_per_epoch = math.ceil(num_batches / gradient_accumulation_steps)

    if gradient_accumulation_steps > 1:
        print(f'Gradient accumulation: {gradient_accumulation_steps} micro-batches per optimizer step')

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets, modal_inputs, visual_inputs = _unpack_batch(batch)
        samples = samples.to(device)
        targets = _move_targets_to_device(targets, device)
        modal_inputs = _move_modal_inputs_to_device(modal_inputs, device)
        visual_inputs = _move_visual_inputs_to_device(visual_inputs, device)
        global_step = epoch * updates_per_epoch + i // gradient_accumulation_steps
        metas = dict(epoch=epoch, step=i, global_step=global_step)
        update_step = (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == num_batches
        accumulation_group_start = (i // gradient_accumulation_steps) * gradient_accumulation_steps
        current_accumulation_steps = min(gradient_accumulation_steps, num_batches - accumulation_group_start)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = _forward_model(
                    model,
                    samples,
                    targets=targets,
                    modal_inputs=modal_inputs,
                    visual_inputs=visual_inputs,
                )
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss / current_accumulation_steps).backward()

            if update_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        else:
            outputs = _forward_model(
                model,
                samples,
                targets=targets,
                modal_inputs=modal_inputs,
                visual_inputs=visual_inputs,
            )
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            (loss / current_accumulation_steps).backward()
            
            if update_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        if update_step:
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        loss_value_scalar = loss_value.item()

        if not math.isfinite(loss_value_scalar):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value_scalar, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value_scalar, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets, modal_inputs, visual_inputs = _unpack_batch(batch)
        samples = samples.to(device)
        targets = _move_targets_to_device(targets, device)
        modal_inputs = _move_modal_inputs_to_device(modal_inputs, device)
        visual_inputs = _move_visual_inputs_to_device(visual_inputs, device)

        outputs = _forward_model(
            model,
            samples,
            modal_inputs=modal_inputs,
            visual_inputs=visual_inputs,
        )

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



