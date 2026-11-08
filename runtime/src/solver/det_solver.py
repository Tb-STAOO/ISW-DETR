"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def _load_best_stat_from_log(self, metric_name, mode='max'):
        best_stat = {'epoch': -1}
        if not self.output_dir:
            return best_stat

        log_path = self.output_dir / 'log.txt'
        if not log_path.exists():
            return best_stat

        metric_key = f'test_{metric_name}'
        try:
            with log_path.open('r') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    values = row.get(metric_key)
                    if not isinstance(values, list) or len(values) == 0:
                        continue
                    value = values[0]
                    if not isinstance(value, (int, float)):
                        continue
                    if metric_name not in best_stat:
                        better = True
                    elif mode == 'max':
                        better = value > best_stat[metric_name]
                    else:
                        better = value < best_stat[metric_name]

                    if better:
                        best_stat['epoch'] = row.get('epoch', -1)
                        best_stat[metric_name] = value
        except OSError:
            return {'epoch': -1}

        return best_stat
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        gradient_accumulation_steps = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
        train_total_batch_size = args.yaml_cfg.get('train_dataloader', {}).get('total_batch_size', None)
        effective_train_batch_size = None
        if train_total_batch_size is not None:
            effective_train_batch_size = train_total_batch_size * gradient_accumulation_steps

        early_stop_cfg = args.yaml_cfg.get('early_stop', {}) or {}
        early_stop_enabled = bool(early_stop_cfg.get('enabled', False))
        early_stop_metric = early_stop_cfg.get('metric', 'coco_eval_bbox')
        early_stop_mode = early_stop_cfg.get('mode', 'max')
        early_stop_patience = int(early_stop_cfg.get('patience', 20))
        early_stop_min_delta = float(early_stop_cfg.get('min_delta', 0.0))
        early_stop_min_epoch = int(early_stop_cfg.get('min_epoch', 0))
        assert early_stop_mode in ['max', 'min'], 'early_stop.mode only supports max or min'

        best_stat = self._load_best_stat_from_log(early_stop_metric, early_stop_mode)
        early_stop_best_value = best_stat.get(early_stop_metric)
        early_stop_best_epoch = best_stat.get('epoch', -1)
        if early_stop_enabled:
            print(
                'Early stopping enabled: '
                f'metric={early_stop_metric}, mode={early_stop_mode}, '
                f'patience={early_stop_patience}, min_delta={early_stop_min_delta}, '
                f'min_epoch={early_stop_min_epoch}, loaded_best={best_stat}'
            )
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            early_stop_bad_epochs = None
            early_stop_metric_value = None
            if early_stop_enabled and early_stop_metric in test_stats:
                values = test_stats[early_stop_metric]
                if isinstance(values, (list, tuple)) and len(values) > 0:
                    early_stop_metric_value = values[0]
                    if early_stop_best_value is None:
                        early_stop_best_value = early_stop_metric_value
                        early_stop_best_epoch = epoch
                    else:
                        if early_stop_mode == 'max':
                            improved = early_stop_metric_value > early_stop_best_value + early_stop_min_delta
                        else:
                            improved = early_stop_metric_value < early_stop_best_value - early_stop_min_delta

                        if improved:
                            early_stop_best_value = early_stop_metric_value
                            early_stop_best_epoch = epoch

                    if early_stop_best_epoch >= 0:
                        early_stop_bad_epochs = max(0, epoch - early_stop_best_epoch)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_train_batch_size': effective_train_batch_size
            }

            if early_stop_enabled:
                log_stats.update({
                    'early_stop_metric': early_stop_metric,
                    'early_stop_metric_value': early_stop_metric_value,
                    'early_stop_best_epoch': early_stop_best_epoch,
                    'early_stop_best_value': early_stop_best_value,
                    'early_stop_bad_epochs': early_stop_bad_epochs,
                    'early_stop_patience': early_stop_patience,
                    'early_stop_min_epoch': early_stop_min_epoch,
                })

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if (
                early_stop_enabled
                and early_stop_bad_epochs is not None
                and epoch >= early_stop_min_epoch
                and early_stop_bad_epochs >= early_stop_patience
            ):
                message = (
                    'Early stopping triggered: '
                    f'epoch={epoch}, metric={early_stop_metric}, '
                    f'best_epoch={early_stop_best_epoch}, '
                    f'best={early_stop_best_value}, '
                    f'current={early_stop_metric_value}, '
                    f'bad_epochs={early_stop_bad_epochs}, '
                    f'patience={early_stop_patience}'
                )
                print(message)
                if self.output_dir and dist_utils.is_main_process():
                    with (self.output_dir / 'early_stop.txt').open('a') as f:
                        f.write(message + '\n')
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
