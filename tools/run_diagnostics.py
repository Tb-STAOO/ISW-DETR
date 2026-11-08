"""Rebuild evaluation-only class/ownership diagnostics and acquisition sessions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'reference_sources')]
sys.dont_write_bytecode = True
from run_private_controls import infer, physical_cache, read, sha, tr
from diagnostic_supervision import attach_supervision
import diagnostics_reference


def write_session_results(controls, protocol, output):
    annotation_path = protocol / 'inputs/holdout.json'
    annotation = read(annotation_path)
    session_by_id = {int(row['id']): row['session_id'] for row in annotation['images']}
    rows = []
    for seed in [42, 3407, 2026]:
        arrays = dict(np.load(controls / 'holdout' / f'seed{seed}/predictions.npz'))
        for session in sorted(set(session_by_id.values())):
            mask = np.asarray([session_by_id[int(image_id)] == session for image_id in arrays['image_ids']])
            if int(mask.sum()) != 40:
                raise ValueError('Expected three acquisition sessions with 40 scenes each')
            subset = {key: arrays[key][mask] for key in ['image_ids', 'boxes', 'original_sizes']}
            for method in ['original_BQR', 'half_readout_original_assignment', 'half_readout_mean_assignment', 'half_readout_max_assignment']:
                metrics = tr.coco_metrics(subset, torch.from_numpy(arrays[method][mask]), annotation_path)
                rows.append(dict(seed=seed, session=session, images=int(mask.sum()), method=method, **metrics))
    (output / 'SESSION_RESULTS.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-root', required=True)
    parser.add_argument('--controls-root', required=True)
    parser.add_argument('--protocol-root', required=True)
    parser.add_argument('--weights-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sessions-only', action='store_true', help='Recompute all four session contrasts in an already generated diagnostics directory')
    args = parser.parse_args()
    torch.set_num_threads(2)
    source, controls = Path(args.inference_root).resolve(), Path(args.controls_root).resolve()
    protocol, output = Path(args.protocol_root).resolve(), Path(args.output).resolve()
    if args.sessions_only:
        if not (output / 'DIAGNOSTICS_QA.json').is_file():
            parser.error('sessions-only requires an existing complete class diagnostics output')
        write_session_results(controls, protocol, output)
        return
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh diagnostics directory')
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / 'prepared'
    prepared.mkdir()
    for seed in [42, 3407, 2026]:
        wrapped, _ = infer.build_model(seed, torch.device('cpu'), str(Path(args.weights_root) / f'detector_seed{seed}.pth'))
        for split in ['internal', 'holdout']:
            arrays = dict(np.load(source / split / f'seed{seed}/predictions/queries.npz'))
            manifest = read(source / 'inputs' / split / 'manifest.json')
            records = []
            for row in manifest['samples']:
                modal, _ = infer.load_transaction(Path(row['transaction']), 'cpu')
                records.append(dict(image_id=int(row['sample_id']), **{key: value[0].tolist() for key, value in modal.items()}))
            data = physical_cache(arrays, records, wrapped.bridge.upstream)
            annotation = read(protocol / 'inputs' / (split + '.json'))
            images = {int(row['id']): row for row in annotation['images']}
            annotations = defaultdict(list)
            for row in annotation['annotations']:
                annotations[int(row['image_id'])].append(row)
            for rows in annotations.values():
                rows.sort(key=lambda row: (int(row['placement_order']), int(row['id'])))
            data['audits'] = [None if item is None else attach_supervision(data['arrays'], index, records[index], item, images, annotations)
                              for index, item in enumerate(data['prepared'])]
            torch.save(data, prepared / f'{split}_seed{seed}.pt')
    tr.OLD_PREPARED = prepared
    diagnostics_reference.ROOT = ROOT
    diagnostics_reference.R21 = controls
    diagnostics_reference.PACKAGE = protocol
    diagnostics_reference.DATA = output
    diagnostics_reference.SOURCE = {}
    diagnostics_reference.MAIN = 'half_readout_mean_assignment'
    diagnostics_reference.SEEDS = (42, 3407, 2026)
    def tracked_read(path):
        diagnostics_reference.SOURCE[str(path)] = sha(path)
        return read(path)
    def write(path, payload):
        Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    diagnostics_reference.read = tracked_read
    diagnostics_reference.sha = sha
    diagnostics_reference.write = write
    diagnostics_reference.append_diagnostic_values = lambda obj: None
    diagnostics = diagnostics_reference.fixed_prediction_diagnostics()
    with np.load(source / 'holdout/seed42/predictions/queries.npz') as query:
        positions = {int(image_id): index for index, image_id in enumerate(query['image_ids'])}
        for case in diagnostics['selected_cases']:
            case['original_rtdetr_argmax'] = int(query['flat_visual_logits'][positions[case['image_id']], case['query_id']].argmax())
    write(output / 'PREDICTION_DIAGNOSTICS.json', diagnostics)
    write_session_results(controls, protocol, output)
    write(output / 'DIAGNOSTICS_QA.json', dict(status='complete', target_matching='Original geometry-only diagnostic matching, applied after RGB/fusion outputs',
          matched_observations=sum(row['foreground'] for row in diagnostics['corrections'] if row['split'] == 'holdout'),
          no_model_training=True, sources=diagnostics_reference.SOURCE))


if __name__ == '__main__':
    main()
