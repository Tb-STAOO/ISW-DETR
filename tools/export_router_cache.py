"""Export acquired clean training queries and attach supervision afterwards.

The reported logit-noise router uses one clean view per source, so unused
RGB perturbation views are unnecessary. Evaluation splits are never exported
as training rows. This is a training input builder, not an AP evaluator.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'reference_sources')]
from run_private_controls import infer,physical_cache,read,sha,data_export


def run(command,log):
    with Path(log).open('w',encoding='utf8') as stream:
        result=subprocess.run([sys.executable,'-X','utf8','-B']+[str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:raise RuntimeError('Export failed; see '+str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset-root','protocol-root','weights-root','validation-inference-root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--seeds',nargs='+',type=int,choices=[42,3407,2026],default=[42,3407,2026])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--smoke-limit',type=int,help='Diagnostic only; full reproduction requires all 420 acquired sources')
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    dataset=Path(args.dataset_root).resolve()/'ScaleVision-MM-V3'
    ann_path=dataset/'annotations/splits/instances_multimodal_train.json'
    original=read(ann_path)
    images=sorted([r for r in original['images'] if 'source_scene_id' not in r],key=lambda r:(r['scene_id'],int(r['id'])))
    if len(images)!=420:raise ValueError('Expected exactly 420 acquired training source images')
    if args.smoke_limit:
        if args.smoke_limit<1 or args.smoke_limit>=420:p.error('smoke-limit must be between 1 and 419')
        images=images[:args.smoke_limit]
    ids={int(r['id']) for r in images}
    subset={**original,'images':images,'annotations':[r for r in original['annotations'] if int(r['image_id']) in ids]}
    annotation=out/'train_sources.json';data_export.write(annotation,subset)
    run([ROOT/'tools/prepare_scalevision.py','--annotations',annotation,'--image-root',dataset/'images','--output',out/'inputs/train'],out/'prepare_train.log')
    hashes={str(ann_path):sha(ann_path)};reports=[]
    for seed in args.seeds:
        checkpoint=Path(args.weights_root).resolve()/f'detector_seed{seed}.pth'
        wrapped,_=infer.build_model(seed,torch.device('cpu'),str(checkpoint));hashes[str(checkpoint)]=sha(checkpoint)
        querydir=out/f'train_seed{seed}'
        run([ROOT/'inference/infer_batch.py','--manifest',out/'inputs/train/manifest.json','--checkpoint',checkpoint,
             '--seed',seed,'--device',args.device,'--batch-size',args.batch_size,'--save-queries','--outdir',querydir],out/f'inference_train_{seed}.log')
        for split in ['train','validation']:
            source=out if split=='train' else Path(args.validation_inference_root).resolve()
            query=querydir/'queries.npz' if split=='train' else source/f'validation/seed{seed}/predictions/queries.npz'
            arrays=dict(np.load(query));arrays['view_ids']=np.zeros(len(arrays['image_ids']),dtype=np.int64)
            manifest=read(source/f'inputs/{split}/manifest.json')
            records=[]
            for r in manifest['samples']:
                modal,_=infer.load_transaction(Path(r['transaction']),'cpu')
                records.append(dict(image_id=int(r['sample_id']),**{k:v[0].tolist() for k,v in modal.items()}))
            if arrays['image_ids'].tolist()!=[int(r['sample_id']) for r in manifest['samples']]:raise ValueError('Input/query order differs')
            annotation_path=annotation if split=='train' else Path(args.protocol_root).resolve()/'inputs/validation.json'
            target=read(annotation_path);imap,amap=data_export.annotation_maps(target)
            if split=='train':
                # Preserve original training exporter query order (ascending
                # selected indices), distinct from legacy evaluation pool order.
                prepared=[data_export.prepare_forward(torch.as_tensor(arrays['visual_logits'][i]),torch.as_tensor(arrays['boxes'][i]),r,wrapped.bridge.upstream)[0] for i,r in enumerate(records)]
                data=dict(arrays=dict(image_ids=arrays['image_ids'],logits=arrays['visual_logits'],base_logits=arrays['flat_visual_logits'],boxes=arrays['boxes'],original_sizes=arrays['original_sizes'],view_ids=arrays['view_ids']),records=records,prepared=prepared)
            else:data=physical_cache(arrays,records,wrapped.bridge.upstream)
            data['arrays']['view_ids']=arrays['view_ids']
            data['audits']=[None if pack is None else data_export.attach_supervision(data['arrays'],i,records[i],pack,imap[int(arrays['image_ids'][i])],amap[int(arrays['image_ids'][i])],'clean') for i,pack in enumerate(data['prepared'])]
            data['metadata']=dict(split='train' if split=='train' else 'validation',clean_only=True,
                inference_device=args.device if split=='train' else 'from supplied validation inference',
                smoke_only=bool(args.smoke_limit),source_annotation_sha256=sha(annotation_path))
            destination=out/'prepared'/f'{split}_seed{seed}.pt';destination.parent.mkdir(exist_ok=True);torch.save(data,destination)
            hashes[str(query)]=sha(query);hashes[str(annotation_path)]=sha(annotation_path)
            reports.append(dict(seed=seed,split=split,images=len(records),cache=str(destination),sha256=sha(destination)))
        print(json.dumps(dict(seed=seed,train_sources=len(images))),flush=True)
    data_export.write(out/'ROUTER_CACHE_MANIFEST.json',dict(status='complete',smoke_only=bool(args.smoke_limit),rows=reports,
        input_sha256=hashes,GT_attached_only_after_inference=True,unused_augmented_views_omitted_for_clean_logit_noise_mode=True))


if __name__=='__main__':main()
