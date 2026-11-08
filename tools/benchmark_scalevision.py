"""Repeat the original balanced five-path batch-1 FP32 timing protocol."""
import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'inference'), str(ROOT / 'reference_sources')]
import infer
from native_benefit_adapter import BenefitRoutedDetector
from native_detector import StateNeutralDetector
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference-root', required=True)
    p.add_argument('--weights-root', required=True)
    p.add_argument('--control-weights-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('This reference timing protocol requires CUDA; CPU is supported by inference, not substituted for this timing table')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.weights_root) / 'detector_seed42.pth'
    new, provenance = infer.build_model(42, torch.device('cuda'), str(checkpoint))
    # build_model disconnects old CVSR for the proposed path. Restore it only
    # on a separate control detector, preserving the native GPU implementations.
    native = copy.deepcopy(new.detector)
    native.cvsr = copy.deepcopy(new.bridge.upstream).cuda().eval()
    bqr_path = Path(args.control_weights_root) / 'bqr/seed42.pth'
    wrapped = BenefitRoutedDetector(native, bqr_path, 42)
    manifest_path = Path(args.inference_root) / 'inputs/validation/manifest.json'
    manifest = infer.read(manifest_path)
    lookup = {int(row['sample_id']): row for row in manifest['samples']}
    scene_lock = ROOT / 'configs/scalevision/benchmark_scenes.json'
    wanted = infer.read(scene_lock)['image_ids']
    if len(wanted) != 18 or len(set(wanted)) != 18:
        raise ValueError('Changed original validation timing scene list')
    samples = []
    hashes = {str(checkpoint): infer.sha(checkpoint), str(bqr_path): infer.sha(bqr_path),
              str(manifest_path): infer.sha(manifest_path), str(scene_lock): infer.sha(scene_lock)}
    for iid in wanted:
        row = lookup[iid]
        image = Image.open(row['image']).convert('RGB')
        modal, record = infer.load_transaction(Path(row['transaction']), 'cuda')
        samples.append((infer.image_tensor(image)[None].cuda(), modal,
                        torch.tensor([[image.width, image.height]], device='cuda')))
        hashes[row['image']] = infer.sha(row['image'])
        hashes[row['transaction']] = infer.sha(row['transaction'])
    post = RTDETRPostProcessor(num_classes=28, num_top_queries=300, remap_mscoco_category=False).cuda().eval()
    names = ['rt_detr_flat', 'identity_state_visual', 'fixed_cvsr', 'full_bqr', 'state_contrast_isw_detr']

    def run(name, sample):
        x, modal, size = sample
        if name == 'rt_detr_flat':
            result = RTDETRTransformerv2.forward(native.decoder, native.encoder(native.backbone(x)), targets=None)
        elif name == 'identity_state_visual':
            result = native.decoder(native.encoder(native.backbone(x)), targets=None)
        elif name == 'fixed_cvsr':
            result = native(x, targets=None, modal_inputs=modal)
        elif name == 'full_bqr':
            result = wrapped(x, targets=None, modal_inputs=modal)
        else:
            result = new(x, targets=None, modal_inputs=modal)
        return post(result, size)

    fallback = []
    x, modal, size = samples[0]
    visual = new.visual(x)
    for kind in ['missing', 'delta_inconsistent']:
        fault = None if kind == 'missing' else copy.deepcopy(modal)
        if fault is not None:
            fault['weight_tokens_raw'][:, :, 2] += 10.
        for name, model in [('fixed_cvsr', native), ('full_bqr', wrapped), ('state_contrast_isw_detr', new)]:
            result = model(x, targets=None, modal_inputs=fault)
            ok = torch.equal(result['pred_logits'], visual['pred_logits_cfqd']) and torch.equal(result['pred_boxes'], visual['pred_boxes'])
            fallback.append(dict(kind=kind, path=name, all_queries_and_boxes_exact_visual=ok))
            if not ok:
                raise RuntimeError('Raw missing/fault fallback check failed: ' + str(fallback[-1]))
    for name in names:
        for i in range(24):
            run(name, samples[i % len(samples)])
    torch.cuda.synchronize()
    rows, memory = [], {}
    for repeat in range(12):
        for j, sample in enumerate(samples):
            shift = (repeat + j) % len(names)
            for name in names[shift:] + names[:shift]:
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = run(name, sample)
                torch.cuda.synchronize()
                rows.append(dict(path=name, repeat=repeat, image_id=wanted[j], ms=1000 * (time.perf_counter() - started)))
                del result
        print(json.dumps(dict(timing_repeat=repeat+1, observations=len(rows))), flush=True)
    for name in names:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = run(name, samples[0])
        torch.cuda.synchronize()
        memory[name] = dict(common_resident_allocated_mib=before/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            increment_over_resident_mib=(torch.cuda.max_memory_allocated()-before)/2**20)
        del result
    total = sum(p.numel() for p in native.parameters())
    aux = sum(p.numel() for n,p in native.named_parameters() if n.startswith('decoder.cfqd_'))
    bqr = sum(p.numel() for p in wrapped.router.parameters())
    params = dict(zip(names, [total-aux, total, total, total+bqr, total]))
    groups = {}
    for name in names:
        values = [r['ms'] for r in rows if r['path'] == name]
        groups[name] = dict(parameters=params[name], median_ms=float(np.median(values)),
            iqr_ms=np.percentile(values,[25,75]).tolist(), p95_ms=float(np.percentile(values,95)), repeats=len(values),
            scene_medians_ms=[dict(image_id=i,median_ms=float(np.median([r['ms'] for r in rows if r['path']==name and r['image_id']==i]))) for i in wanted], **memory[name])
    report = dict(status='complete', device=torch.cuda.get_device_name(), precision='FP32', batch_size=1,
        seed=42, scene_count=len(samples), image_ids=wanted, warmup_per_path=24, repeats_per_scene=12,
        validation_only=True, cyclic_balanced_order=True, results=groups, IS_auxiliary_parameters=aux, BQR_parameters=bqr,
        new_rule_trainable_parameters=sum(p.numel() for p in new.bridge.parameters()), all_model_parameters_frozen=True,
        timing_includes=['visual forward','physical validation and fusion','inference device transfers','common top300 postprocessing','CUDA synchronization'],
        timing_excludes=['image decoding/resizing','initial input transfer','sensor waiting'],
        new_path_numeric_backend='GPU visual, CPU FP32 fusion, GPU postprocessor',
        other_fusion_paths_numeric_backend='native GPU fusion',
        memory_note='All models and input samples are resident; increment is over shared resident baseline, not isolated model footprint.',
        cpu_threads=2, software=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__),
        input_hashes=hashes, checkpoint_provenance=provenance, fallback_checks=fallback,
        original_timing_values_copied=False, fresh_training=False)
    if any(infer.sha(path) != h for path,h in hashes.items()):
        raise RuntimeError('An input changed during timing')
    infer.write(out/'COMPUTATIONAL_SUMMARY.json',report)
    infer.write(out/'repeat_times.json',rows)
    print(json.dumps(dict(status='complete',parameters=params,median_ms={k:v['median_ms'] for k,v in groups.items()})),flush=True)


if __name__ == '__main__':
    main()
