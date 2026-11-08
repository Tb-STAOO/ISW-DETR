"""Run complete fixed-checkpoint ScaleVision splits with separate COCO scoring."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def command(args, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B'] + [str(v) for v in args],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Execution failed; inspect ' + str(logfile))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-root', required=True, help='Directory containing inputs/{validation,internal,holdout}.json')
    parser.add_argument('--dataset-root', required=True, help='Root containing ScaleVision-MM-V3 and ScaleVision-MM-External-120')
    parser.add_argument('--weights-root', required=True)
    parser.add_argument('--output', required=True, help='Fresh reproduction directory')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8, help='Paper accuracy protocol uses 8 with a real tail batch')
    parser.add_argument('--splits', nargs='+', choices=['validation', 'internal', 'holdout'], default=['validation', 'internal', 'holdout'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh reproduction directory')
    output.mkdir(parents=True, exist_ok=True)
    protocol = Path(args.protocol_root).resolve()
    data = Path(args.dataset_root).resolve()
    weights = Path(args.weights_root).resolve()
    rows = []
    for split in args.splits:
        image_root = data / ('ScaleVision-MM-External-120' if split == 'holdout' else 'ScaleVision-MM-V3') / 'images'
        annotation = protocol / 'inputs' / (split + '.json')
        inputs = output / 'inputs' / split
        command([ROOT / 'tools/prepare_scalevision.py', '--annotations', annotation,
                 '--image-root', image_root, '--output', inputs], output / 'logs' / f'prepare_{split}.log')
        for seed in args.seeds:
            run = output / split / f'seed{seed}'
            command([ROOT / 'inference/infer_batch.py', '--manifest', inputs / 'manifest.json',
                     '--checkpoint', weights / f'detector_seed{seed}.pth', '--seed', seed,
                     '--device', args.device, '--batch-size', args.batch_size, '--save-queries',
                     '--outdir', run / 'predictions'], output / 'logs' / f'inference_{split}_{seed}.log')
            for field, method in [('flat_visual_detections', 'RT-DETRv2'),
                                  ('visual_detections', 'I-S visual'), ('detections', 'ISW-DETR')]:
                evaluation = run / 'evaluation' / field
                command([ROOT / 'tools/evaluate_scalevision.py', '--annotations', annotation,
                         '--predictions', run / 'predictions', '--field', field, '--output', evaluation],
                        output / 'logs' / f'evaluation_{split}_{seed}_{field}.log')
                result = read(evaluation / 'METRICS.json')
                rows.append(dict(split=split, seed=seed, method=method, **result['metrics_percent']))
            print(json.dumps(dict(split=split, seed=seed, metrics=rows[-3:])), flush=True)
            (output / 'PARTIAL_RESULTS.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    summary = []
    for split in args.splits:
        for method in ['RT-DETRv2', 'I-S visual', 'ISW-DETR']:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            values = [r['AP'] for r in selected]
            summary.append(dict(split=split, method=method, seeds=[r['seed'] for r in selected],
                                AP_mean=statistics.mean(values), AP_sample_SD=statistics.stdev(values) if len(values) > 1 else None))
    result = dict(status='complete', scope='Full-split inference and COCO evaluation from supplied trained checkpoints',
                  full_original_training_reproduced=False, batch_size=args.batch_size, rows=rows, summary=summary)
    (output / 'SCALEVISION_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
