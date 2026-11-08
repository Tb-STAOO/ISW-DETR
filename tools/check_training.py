"""One-epoch/eight-image installation smoke check for the three public branches.

This deliberately shortened run checks the real training, loss, validation and
checkpoint path. Its metrics are not research results or a training reproduction.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml
import package_paths

ROOT = package_paths.ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True, help='Official COCO initialization for base')
    parser.add_argument('--base-checkpoint', required=True, help='Published base_seed42/best.pth for flat/is smoke checks')
    parser.add_argument('--output', required=True, help='Fresh diagnostic directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh diagnostic directory')
    output.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_root).resolve()
    package_paths.DATA = data
    for split in ['train', 'dev']:
        annotation = read(data / 'annotations' / (split + '.json'))
        annotation['images'] = sorted(annotation['images'], key=lambda r: int(r['id']))[:8]
        ids = {r['id'] for r in annotation['images']}
        annotation['annotations'] = [r for r in annotation['annotations'] if r['image_id'] in ids]
        (output / (split + '.json')).write_text(json.dumps(annotation), encoding='utf-8')
    rows = []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for variant in ['base', 'flat', 'is']:
        config = yaml.safe_load(package_paths.local_config(variant, 42, output).read_text(encoding='utf-8'))
        config.update(epoches=1, print_freq=1, checkpoint_freq=1, output_dir=str(output / variant),
                      use_amp=False, summary_dir=None)
        config['scaler']['enabled'] = False
        for loader, split in [('train_dataloader', 'train'), ('val_dataloader', 'dev')]:
            config[loader]['dataset']['ann_file'] = str(output / (split + '.json'))
            config[loader]['dataset']['img_folder'] = str(data)
            config[loader]['num_workers'] = 0
        path = output / (variant + '_smoke.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        initializer = Path(args.pretrained if variant == 'base' else args.base_checkpoint).resolve(strict=True)
        command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / 'runtime/tools/train.py'),
                   '-c', str(path), '--seed', '42', '-d', args.device, '-t', str(initializer)]
        with (output / (variant + '_console.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT / 'runtime', env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Training smoke check failed: ' + str(output / (variant + '_console.log')))
        logs = [json.loads(line) for line in (output / variant / 'log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(logs) != 1:
            raise ValueError('Expected exactly one diagnostic epoch')
        losses = {name: value for name, value in logs[0].items() if name.startswith('train_loss')}
        if not losses or not all(math.isfinite(float(value)) for value in losses.values()):
            raise ValueError('Missing/nonfinite diagnostic losses')
        if not (output / variant / 'best.pth').is_file():
            raise ValueError('Best checkpoint was not saved')
        rows.append(dict(variant=variant, passed=True, losses=losses,
                         parameters=logs[0].get('n_parameters'), checkpoint_saved=True))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(passed=True, scope='One diagnostic epoch on eight training/eight validation images; not paper metrics',
                  device=args.device, variants=rows)
    (output / 'TRAINING_SMOKE_QA.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
