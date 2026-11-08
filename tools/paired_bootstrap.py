"""Paired scene bootstrap of whole-sample COCO AP for the three fixed seeds."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'reference_sources'))
import exact_coco_ap as exact

METHODS = ('half_readout_mean_assignment', 'original_BQR',
           'half_readout_original_assignment', 'half_readout_max_assignment')
SEEDS = (42, 3407, 2026)
CONTEXT = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records_for(arrays, method):
    import torch
    sys.path.insert(0, str(ROOT / 'inference/runtime'))
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    torch.set_num_threads(1)
    post = RTDETRPostProcessor(num_classes=28, num_top_queries=300, remap_mscoco_category=False).eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(arrays['image_ids']), 64):
            output = post(dict(pred_logits=torch.as_tensor(arrays[method][start:start + 64]),
                               pred_boxes=torch.as_tensor(arrays['boxes'][start:start + 64])),
                          torch.as_tensor(arrays['original_sizes'][start:start + 64]))
            for image_id, row in zip(arrays['image_ids'][start:start + 64], output):
                for box, score, category in zip(row['boxes'].tolist(), row['scores'].tolist(), row['labels'].tolist()):
                    x1, y1, x2, y2 = box
                    records.append(dict(image_id=int(image_id), category_id=int(category), score=float(score),
                                        bbox=[x1, y1, max(0., x2 - x1), max(0., y2 - y1)]))
    return records


def initialize(path):
    global CONTEXT
    with Path(path).open('rb') as stream:
        CONTEXT = pickle.load(stream)


def chunk(draws):
    values = np.zeros((len(draws), len(METHODS)))
    for index, draw in enumerate(draws):
        for column, method in enumerate(METHODS):
            values[index, column] = np.mean([exact.ap(CONTEXT[seed][method], draw) for seed in SEEDS])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-root', required=True, help='Completed run_private_controls.py output')
    parser.add_argument('--annotations', required=True, help='Fixed 120-scene holdout annotation JSON')
    parser.add_argument('--output', required=True, help='Fresh statistical output directory')
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--random-seed', type=int, default=2026102105)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=25)
    args = parser.parse_args()
    if min(args.replicates, args.workers, args.chunk_size) < 1:
        parser.error('Replicates/workers/chunk-size must be positive')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh statistical output directory')
    output.mkdir(parents=True, exist_ok=True)
    dataset = read(args.annotations)
    ids = sorted(int(row['id']) for row in dataset['images'])
    if len(ids) != 120 or len(set(ids)) != 120:
        raise ValueError('The manuscript protocol requires all 120 measured scenes')
    draws = np.random.default_rng(args.random_seed).integers(0, len(ids), size=(args.replicates, len(ids)), dtype=np.int16)
    np.save(output / 'scene_draws.npy', draws)
    context, points, validation = {}, {}, []
    hashes = {str(Path(args.annotations).resolve()): sha(args.annotations),
              str(Path(__file__).resolve()): sha(__file__),
              str(ROOT / 'reference_sources/exact_coco_ap.py'): sha(ROOT / 'reference_sources/exact_coco_ap.py')}
    for seed in SEEDS:
        context[seed] = {}
        path = Path(args.controls_root).resolve() / 'holdout' / f'seed{seed}/predictions.npz'
        hashes[str(path)] = sha(path)
        with np.load(path) as source:
            arrays = dict(source)
        if arrays['image_ids'].tolist() != ids:
            raise ValueError('Query IDs differ from the complete fixed scene order')
        for method in METHODS:
            records = records_for(arrays, method)
            evaluated = exact.coco_eval(dataset, records)
            cached = exact.extract(evaluated)
            point = exact.ap(cached, np.arange(len(ids)))
            if abs(point - evaluated.stats[0]) > 1e-12:
                raise RuntimeError('Cached AP differs from fresh COCO evaluation')
            context[seed][method] = cached
            points[f'{seed}:{method}'] = point
            if method in METHODS[:2]:
                for name, draw in [('reverse', np.arange(len(ids))[::-1]),
                                   ('allzero', np.zeros(len(ids), dtype=int)), ('first_random', draws[0])]:
                    copied, predictions = exact.duplicate(dataset, records, ids, draw)
                    fresh = float(exact.coco_eval(copied, predictions).stats[0])
                    computed = exact.ap(cached, draw)
                    if abs(fresh - computed) > 1e-12:
                        raise RuntimeError('Resampled AP cache failed an independent duplicated-image COCO check')
                    validation.append(dict(seed=seed, method=method, draw=name, absolute_error=abs(fresh - computed)))
            print('PREPARED', seed, method, flush=True)
    with (output / 'context.pkl').open('wb') as stream:
        pickle.dump(context, stream, pickle.HIGHEST_PROTOCOL)
    (output / 'EXACT_COCO_QA.json').write_text(json.dumps(dict(passed=True, checks=validation, points=points), indent=2) + '\n', encoding='utf-8')
    protocol = dict(replicates=args.replicates, random_seed=args.random_seed, scene_count=len(ids),
                    training_seeds=list(SEEDS), methods=list(METHODS),
                    estimand='Difference of mean whole-sample AP across three fixed model seeds; same sampled scene indices for every method and seed',
                    seed_resampling=False, interval='Pointwise percentile 95%, NumPy linear 2.5/97.5',
                    source_hashes=hashes, workers=args.workers, chunk_size=args.chunk_size)
    (output / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    means = np.zeros((args.replicates, len(METHODS)))
    started = time.monotonic()
    complete = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(output / 'context.pkl'),)) as pool:
        jobs = {pool.submit(chunk, draws[start:start + args.chunk_size]): start for start in range(0, len(draws), args.chunk_size)}
        for job in as_completed(jobs):
            start = jobs[job]
            values = job.result()
            means[start:start + len(values)] = values
            complete += len(values)
            if complete % 500 == 0 or complete == args.replicates:
                print('BOOTSTRAP', complete, '/', args.replicates, 'seconds', round(time.monotonic() - started, 1), flush=True)
                (output / 'PROGRESS.json').write_text(json.dumps(dict(complete=complete, total=args.replicates, elapsed_seconds=time.monotonic() - started)), encoding='utf-8')
    differences = 100 * (means[:, :1] - means[:, 1:])
    np.save(output / 'resampled_mean_aps.npy', means)
    np.save(output / 'paired_differences.npy', differences)
    contrasts = []
    for column, other in enumerate(METHODS[1:]):
        per_seed = [100 * (points[f'{seed}:{METHODS[0]}'] - points[f'{seed}:{other}']) for seed in SEEDS]
        contrasts.append(dict(comparison=METHODS[0] + ' minus ' + other,
                              point_gain_ap_points=float(np.mean(per_seed)),
                              per_seed_point_gain_ap_points=per_seed,
                              percentile95_ci_ap_points=np.percentile(differences[:, column], [2.5, 97.5]).tolist()))
    if any(sha(path) != value for path, value in hashes.items()):
        raise ValueError('A source/input changed while computing the bootstrap')
    summary = dict(status='complete', **protocol, contrasts=contrasts,
                   elapsed_seconds=time.monotonic() - started,
                   caveat='Conditional scene resampling does not remove model/rule selection bias or create independent physical-unit data')
    (output / 'BOOTSTRAP_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(contrasts), flush=True)


if __name__ == '__main__':
    main()
