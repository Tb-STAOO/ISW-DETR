"""Reproduce locked validation-only state/mass interventions and score controls."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'reference_sources')]
from run_private_controls import infer, physical_cache, proposals, read, sha, tr
from calibration_reference import alternatives, pack_with_z
from state_controls import changed_pack, fixed_logits
from stress_reference import METHODS, infer as stress_infer, with_visual_and_evidence
from visual_interventions import perturbations
from physical_interventions import evidence_pack


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-root', required=True)
    parser.add_argument('--protocol-root', required=True)
    parser.add_argument('--weights-root', required=True)
    parser.add_argument('--control-weights-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh analysis directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = Path(args.inference_root).resolve()
    controls = Path(args.control_weights_root).resolve()
    annotation = Path(args.protocol_root).resolve() / 'inputs/validation.json'
    mapping = torch.tensor(read(ROOT / 'inference/inputs/taxonomy.json')['leaf_to_parent'])
    visual_rows, physical_rows, calibration_rows, readout_rows, gate_checks = [], [], [], [], []
    hashes = {str(annotation): sha(annotation)}
    for seed in [42, 3407, 2026]:
        query_path = source / 'validation' / f'seed{seed}/predictions/queries.npz'
        manifest = read(source / 'inputs/validation/manifest.json')
        records = []
        for row in manifest['samples']:
            modal, _ = infer.load_transaction(Path(row['transaction']), 'cpu')
            records.append(dict(image_id=int(row['sample_id']), **{key: value[0].tolist() for key, value in modal.items()}))
        arrays = dict(np.load(query_path))
        wrapped, _ = infer.build_model(seed, torch.device('cpu'), str(Path(args.weights_root) / f'detector_seed{seed}.pth'))
        data = physical_cache(arrays, records, wrapped.bridge.upstream)
        packed = tr.pack(data)
        checkpoint = controls / f'bqr/seed{seed}.pth'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        router = tr.model_class()(**payload['config']).eval()
        router.load_state_dict(payload['state_dict'])
        hashes[str(query_path)] = sha(query_path)
        hashes[str(checkpoint)] = sha(checkpoint)
        for name, values in alternatives(packed['z'], mapping).items():
            changed = pack_with_z(packed, values, reassign=not name.endswith('_fixed'))
            logits = fixed_logits(data, changed)
            calibration_rows.append(dict(seed=seed, method=name, **tr.coco_metrics(data['arrays'], logits, annotation)))
        mean = changed_pack(packed, mapping, 'mean_erase')
        contrast = packed['z'] - mean['z']
        physical = 2 * packed['scene'][:, None] + .2 * mean['query_record_residual']
        for alpha in (0., .5, 1.):
            for gate in (.5, 1., 1.5):
                changed = {key: value.clone() for key, value in packed.items()}
                changed['base'] = packed['z'] - alpha * contrast + gate * physical
                logits = fixed_logits(data, changed)
                readout_rows.append(dict(seed=seed, alpha=alpha, gate=gate,
                                         key=f'alpha{alpha:g}_gate{gate:g}', **tr.coco_metrics(data['arrays'], logits, annotation)))
        for condition, (visual, temperature) in perturbations(data, packed, mapping).items():
            changed_input = with_visual_and_evidence(packed, visual, temperature)
            for method in METHODS:
                logits, _, _ = stress_infer(data, changed_input, mapping, method, router, payload['inference_scale'])
                visual_rows.append(dict(seed=seed, method=method, condition=condition,
                                        **tr.coco_metrics(data['arrays'], logits, annotation)))
            print('VISUAL', seed, condition, flush=True)
            write(output / 'VISUAL_STRESS_RESULTS.json', dict(rows=visual_rows))
        for sigma in (0., .25, 1., 5.):
            changed, metadata = evidence_pack(data, packed, wrapped.bridge.upstream, sigma)
            for method in METHODS:
                logits, _, _ = stress_infer(data, changed, mapping, method, router, payload['inference_scale'])
                physical_rows.append(dict(seed=seed, method=method, sigma_g=sigma,
                                          **tr.coco_metrics(data['arrays'], logits, annotation)))
            gate_checks.append(dict(seed=seed, sigma_g=sigma, images=len(metadata),
                                    all_accepted=all(row['gate'] for row in metadata), records=metadata))
            print('PHYSICAL', seed, sigma, flush=True)
            write(output / 'PHYSICAL_STRESS_RESULTS.json', dict(rows=physical_rows, gate_checks=gate_checks))
        write(output / 'CALIBRATION_CONTROL_RESULTS.json', dict(rows=calibration_rows))
        write(output / 'REASSOCIATED_SCREEN_RESULTS.json', dict(rows=readout_rows))
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Inputs changed during analysis')
    write(output / 'ANALYSIS_QA.json', dict(status='complete', source_hashes=hashes,
          visual_configurations=len(visual_rows), physical_configurations=len(physical_rows),
          calibration_configurations=len(calibration_rows), readout_configurations=len(readout_rows),
          original_intervention_function_AST_preserved=True, model_training=False))


if __name__ == '__main__':
    main()
