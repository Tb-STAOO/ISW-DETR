"""Rebuild physical inputs from fresh RGB queries and replay frozen controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'inference'), str(ROOT / 'reference_sources')]
import infer
import train_benefit_router as tr
import data_export
import state_controls
from transforms import transform
from e1_core import ConfidencePredictionRouter, map_logits


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@torch.no_grad()
def physical_cache(arrays, records, module, visual_key='visual_logits'):
    logits = torch.as_tensor(arrays[visual_key], dtype=torch.float32)
    boxes = torch.as_tensor(arrays['boxes'], dtype=torch.float32)
    prepared = []
    for index, record in enumerate(records):
        item, _ = data_export.prepare_forward(logits[index], boxes[index], record, module)
        if item is not None:
            order = module._select_candidates(logits[index], boxes[index], len(item['candidates']))
            lookup = {int(query): j for j, query in enumerate(item['candidates'])}
            indices = torch.tensor([lookup[int(query)] for query in order])
            for key in ['z', 'boxes', 'base']:
                item[key] = item[key][indices]
            item['candidates'] = order
            # Preserve the original cache interface. Assignment is recomputed
            # by the unchanged original pack() in this actual candidate order.
            item = {key: item[key] for key in ['z', 'boxes', 'base', 'scene', 'logp', 'raw', 'residual', 'candidates', 'valid']}
        prepared.append(item)
    return dict(arrays=dict(image_ids=arrays['image_ids'], logits=logits.numpy(),
                            base_logits=arrays['flat_visual_logits'], boxes=arrays['boxes'],
                            original_sizes=arrays['original_sizes']),
                records=records, prepared=prepared, audits=[None] * len(records))


def proposals(data, mapping):
    packed = tr.pack(data)
    mean = state_controls.changed_pack(packed, mapping, 'mean_erase')
    maximum = state_controls.changed_pack(packed, mapping, 'max_erase')
    outputs = dict(original_fixed_CVSR=state_controls.fixed_logits(data, packed),
                   full_mean_erasure=state_controls.fixed_logits(data, mean),
                   full_max_erasure=state_controls.fixed_logits(data, maximum))
    half = packed['z'] - .5 * (packed['z'] - transform(packed['z'], mapping, 'mean_erase'))
    for name, association in [('original', packed), ('mean', mean), ('max', maximum)]:
        changed = {key: value.clone() for key, value in association.items()}
        changed['base'] = half + (2 * packed['scene'][:, None] + .2 * association['query_record_residual'])
        outputs['half_readout_' + name + '_assignment'] = state_controls.fixed_logits(data, changed)
    return outputs, packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-root', required=True, help='Completed run_scalevision.py output including all queries')
    parser.add_argument('--protocol-root', required=True)
    parser.add_argument('--weights-root', required=True)
    parser.add_argument('--control-weights-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--splits', nargs='+', choices=['validation', 'internal', 'holdout'], default=['validation', 'internal', 'holdout'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 3407, 2026], default=[42, 3407, 2026])
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = Path(args.inference_root).resolve()
    controls = Path(args.control_weights_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    mapping = torch.tensor(read(ROOT / 'inference/inputs/taxonomy.json')['leaf_to_parent'])
    alpha = float(read(controls / 'map_selection.json')['alpha'])
    rows, input_hashes = [], {}
    for seed in args.seeds:
        checkpoint = Path(args.weights_root).resolve() / f'detector_seed{seed}.pth'
        wrapped, _ = infer.build_model(seed, torch.device('cpu'), str(checkpoint))
        bqr_path, confidence_path = controls / f'bqr/seed{seed}.pth', controls / f'confidence/seed{seed}.pth'
        bqr_payload = torch.load(bqr_path, map_location='cpu', weights_only=True)
        bqr = tr.model_class()(**bqr_payload['config']).eval()
        bqr.load_state_dict(bqr_payload['state_dict'])
        confidence_payload = torch.load(confidence_path, map_location='cpu', weights_only=True)
        confidence = ConfidencePredictionRouter(**confidence_payload['config']).eval()
        confidence.load_state_dict(confidence_payload['state_dict'])
        for path in [checkpoint, bqr_path, confidence_path, controls / 'map_selection.json']:
            input_hashes[str(path)] = sha(path)
        for split in args.splits:
            query_path = source / split / f'seed{seed}/predictions/queries.npz'
            manifest = read(source / 'inputs' / split / 'manifest.json')
            records = [read(row['transaction']) for row in manifest['samples']]
            arrays = dict(np.load(query_path))
            if arrays['image_ids'].tolist() != [int(row['sample_id']) for row in manifest['samples']]:
                raise ValueError('Query and observed-input ordering differ')
            data = physical_cache(arrays, records, wrapped.bridge.upstream)
            outputs, packed = proposals(data, mapping)
            outputs['rt_detr_visual'] = torch.as_tensor(arrays['flat_visual_logits'])
            outputs['identity_visual'] = torch.as_tensor(arrays['visual_logits'])
            gaussian = outputs['identity_visual'].clone()
            for index, item in enumerate(data['prepared']):
                if item is not None:
                    gaussian[index] += 1.5 * item['scene']
            outputs['matched_gaussian'] = gaussian
            outputs['original_BQR'] = tr.replay(bqr, data, packed, 'cpu', scale=bqr_payload['inference_scale'])[0]
            outputs['confidence_prediction_adapted'] = tr.replay(confidence, data, packed, 'cpu', scale=confidence_payload['inference_scale'])[0]
            outputs['joint_map_adapted'] = map_logits(data, alpha)[0]
            flat_data = physical_cache(arrays, records, wrapped.bridge.upstream, 'flat_visual_logits')
            outputs['no_IS_new'] = proposals(flat_data, mapping)[0]['half_readout_mean_assignment']
            if not torch.equal(outputs['half_readout_mean_assignment'], torch.as_tensor(arrays['isw_logits'])):
                raise RuntimeError('Rebuilt physical refinement differs from the complete raw-input RGB inference')
            destination = output / split / f'seed{seed}'
            destination.mkdir(parents=True)
            np.savez_compressed(destination / 'predictions.npz', image_ids=arrays['image_ids'],
                                boxes=arrays['boxes'], original_sizes=arrays['original_sizes'],
                                **{name: value.numpy() for name, value in outputs.items()})
            annotation = Path(args.protocol_root).resolve() / 'inputs' / (split + '.json')
            for method, logits in outputs.items():
                metrics = tr.coco_metrics(data['arrays'], logits, annotation)
                rows.append(dict(split=split, seed=seed, method=method, **metrics))
            input_hashes[str(query_path)] = sha(query_path)
            input_hashes[str(annotation)] = sha(annotation)
            print(json.dumps(dict(split=split, seed=seed, methods=len(outputs), complete_rows=len(rows))), flush=True)
            (output / 'PARTIAL_CONTROLS.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    result = dict(status='complete', scope='All frozen private comparators reconstructed from fresh RGB query outputs and raw measurements',
                  no_annotation_forward=True, input_hashes=input_hashes, rows=rows,
                  early_original_training_reproduced=False)
    (output / 'PRIVATE_CONTROLS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
