"""Recompute all nine public checkpoints x seven scores and compare references."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights-root', required=True, help='Directory containing base/flat/is_seed*/best.pth')
    parser.add_argument('--output', required=True, help='Fresh evaluation root')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--existing-result', action='append', default=[], help='Optional previously verified full-split PACKAGED_RESULTS.json')
    parser.add_argument('--reference', default=str(ROOT / 'configs/cccm/reference_results.json'))
    parser.add_argument('--reference-policy', choices=['strict','report'], default='strict',
                        help='strict requires exact reference metrics; report retains all differences for a new training/environment run')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    existing = {}
    for path in args.existing_result:
        payload = read(path)
        key = (payload['visual_variant'], payload['seed'])
        if key in existing:
            parser.error('Duplicate existing-result configuration')
        existing[key] = (Path(path).resolve(), payload)
    reference = read(args.reference)
    expected = {(row['visual_variant'], row['training_seed'], row['method']): row['metrics'] for row in reference['rows']}
    all_rows = []
    source_results = []
    comparison = []
    for variant in ['base', 'flat', 'is']:
        for seed in [42, 3407, 2026]:
            checkpoint = Path(args.weights_root).resolve() / f'{variant}_seed{seed}/best.pth'
            if (variant, seed) in existing:
                path, payload = existing[variant, seed]
                if payload['images'] != 2969 or payload['split'] != 'test' or payload['status'] != 'complete':
                    raise ValueError('Existing result is not the complete public test split')
                if sha(checkpoint) not in payload['source_lock']['source_sha256'].values():
                    raise ValueError('Existing result checkpoint does not match the supplied weights')
            else:
                run = output / f'{variant}_seed{seed}'
                command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / 'tools/run_cccm.py'), 'evaluate',
                           '--variant', variant, '--seed', str(seed), '--data-root', str(Path(args.data_root).resolve()),
                           '--checkpoint', str(checkpoint), '--device', args.device,
                           '--batch-size', str(args.batch_size), '--output', str(run)]
                with (output / f'{variant}_seed{seed}_console.log').open('w', encoding='utf-8') as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('Public evaluation failed: ' + str(run))
                path = run / 'PACKAGED_RESULTS.json'
                payload = read(path)
            if len(payload['metrics']) != 7:
                raise ValueError('Expected all seven scoring variants')
            source_results.append(dict(variant=variant, seed=seed, path=str(path), sha256=sha(path),
                                       checkpoint_sha256=sha(checkpoint)))
            for method, metrics in payload['metrics'].items():
                old = expected[variant, seed, method]
                fields = [key for key in metrics if isinstance(metrics[key], (float, int)) and key in old and isinstance(old[key], (float, int))]
                differences = {key: abs(metrics[key] - old[key]) for key in fields}
                comparison.append(dict(variant=variant, seed=seed, method=method, fields=len(fields),
                                       max_absolute_delta=max(differences.values())))
                all_rows.append(dict(visual_variant=variant, training_seed=seed, method=method, metrics=metrics))
            progress = dict(status='in_progress', configurations=len(all_rows), rows=all_rows,
                            source_results=source_results, reference_comparison=comparison)
            (output / 'PARTIAL_PUBLIC_RESULTS.json').write_text(json.dumps(progress, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(dict(variant=variant, seed=seed, configurations=len(all_rows),
                                  max_metric_delta=max(row['max_absolute_delta'] for row in comparison))), flush=True)
    report = dict(status='complete', scope='Full public test inference and all seven scores from nine supplied checkpoints',
                  fresh_training_reproduced=False, total_configurations=len(all_rows),
                  reference_sha256=sha(args.reference), reference_comparison=comparison,
                  reference_policy=args.reference_policy,
                  all_metrics_identical=all(row['max_absolute_delta'] == 0 for row in comparison),
                  source_results=source_results, rows=all_rows)
    (output / 'PUBLIC_RESULTS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.reference_policy=='strict' and not report['all_metrics_identical']:
        raise RuntimeError('Evaluation finished but reference differences require review; see PUBLIC_RESULTS.json')


if __name__ == '__main__':
    main()
