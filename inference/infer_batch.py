"""Ordered RGB/transaction manifest inference with real, unpadded tail batches."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from infer import ROOT,resolve,read,write,sha,build_model,load_transaction,image_tensor,prediction_payload

SAMPLE_KEYS={'image','transaction','sample_id','source_image_id_provenance_only','labels_or_boxes_in_input'}

def load_manifest(path):
    """Paths resolve against package root, exactly as in infer.py."""
    raw=read(path)
    if isinstance(raw,dict):
        extra=set(raw)-{'schema_version','samples'}
        if extra:raise ValueError(f'Unknown manifest keys; GT is forbidden: {sorted(extra)}')
        raw=raw.get('samples')
    if not isinstance(raw,list) or not raw:raise ValueError('Manifest must contain a nonempty samples list.')
    samples=[]
    for i,row in enumerate(raw):
        if not isinstance(row,dict):raise ValueError(f'Sample {i} must be a JSON object.')
        extra=set(row)-SAMPLE_KEYS
        if extra:raise ValueError(f'Unknown sample keys; GT is forbidden: {sorted(extra)}')
        if 'labels_or_boxes_in_input' in row and row['labels_or_boxes_in_input'] is not False:
            raise ValueError('labels_or_boxes_in_input may only be false; GT is not accepted.')
        if not isinstance(row.get('image'),str) or not row['image']:
            raise ValueError(f'Sample {i} requires an image path string.')
        tx=row.get('transaction')
        if tx is not None and (not isinstance(tx,str) or not tx):
            raise ValueError(f'Sample {i} transaction must be a path string or null.')
        if 'sample_id' in row and (not isinstance(row['sample_id'],(str,int)) or isinstance(row['sample_id'],bool)):
            raise ValueError('sample_id must be a string or integer when supplied.')
        if 'source_image_id_provenance_only' in row and (not isinstance(row['source_image_id_provenance_only'],(str,int)) or isinstance(row['source_image_id_provenance_only'],bool)):
            raise ValueError('source_image_id_provenance_only must be a string or integer, not a nested object.')
        ip=resolve(row['image']);tp=None if tx is None else resolve(tx)
        if not ip.is_file():raise FileNotFoundError(ip)
        if tp is not None and not tp.is_file():raise FileNotFoundError(tp)
        modal,record=(None,None) if tp is None else load_transaction(tp,'cpu')
        samples.append(dict(index=i,sample_id=row.get('sample_id',i),image=ip,transaction=tp,
            modal=modal,record=record,source_image_id_provenance_only=row.get('source_image_id_provenance_only')))
    return samples

def collate_modal(modals):
    """Pad event dimensions only. RGB samples are never duplicated or padded.

    Missing transactions receive a false event mask; their final logits must
    equal their own visual logits. Scalar defaults follow native CVSR semantics.
    """
    if not modals or all(m is None for m in modals):return None
    batch=len(modals)
    events=[0 if m is None else int(m['weight_tokens_raw'].shape[1]) for m in modals]
    maximum=max(events,default=0)
    raw=torch.zeros(batch,maximum,7,dtype=torch.float32)
    mask=torch.zeros(batch,maximum,dtype=torch.bool)
    order=torch.zeros(batch,maximum,dtype=torch.float32)
    total=torch.full((batch,),float('nan'),dtype=torch.float32)
    baseline=torch.full((batch,),float('nan'),dtype=torch.float32)
    available=torch.zeros(batch,dtype=torch.bool)
    independent=torch.zeros(batch,dtype=torch.bool)
    def scalar(m,key,default):
        if key not in m:return default
        value=m[key].detach().cpu()
        if value.numel()!=1:raise ValueError(f'{key} must be a scalar for each sample.')
        return value.item()
    for i,(m,e) in enumerate(zip(modals,events)):
        if m is None:continue
        if tuple(m['weight_tokens_raw'].shape)!=(1,e,7):raise ValueError('Each transaction must be [1,E,7].')
        raw[i,:e]=m['weight_tokens_raw'][0].detach().cpu()
        if 'weight_mask' in m:
            if tuple(m['weight_mask'].shape)!=(1,e):raise ValueError('weight_mask must be [1,E].')
            mask[i,:e]=m['weight_mask'][0].detach().cpu()
        else:mask[i,:e]=True
        if 'event_order' in m:
            if tuple(m['event_order'].shape)!=(1,e):raise ValueError('event_order must be [1,E].')
            order[i,:e]=m['event_order'][0].detach().cpu()
        else:order[i,:e]=torch.arange(e,dtype=torch.float32)
        if 'scene_total_weight_g' in m and 'empty_baseline_g' in m:
            total[i]=scalar(m,'scene_total_weight_g',float('nan'))
            baseline[i]=scalar(m,'empty_baseline_g',float('nan'))
        finite=bool(torch.isfinite(total[i])&torch.isfinite(baseline[i]))
        available[i]=bool(scalar(m,'scene_total_available',finite))
        independent[i]=bool(scalar(m,'scene_total_independent',False))
    return dict(weight_tokens_raw=raw,weight_mask=mask,event_order=order,
        scene_total_weight_g=total,empty_baseline_g=baseline,
        scene_total_available=available,scene_total_independent=independent)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--seed',type=int,choices=(42,3407,2026),default=42)
    parser.add_argument('--checkpoint',help='External trained checkpoint; weights are not included in this code-only package')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--outdir',default='outputs/batch')
    parser.add_argument('--save-queries',action='store_true',help='Save all query arrays for separate evaluation/statistical reproduction')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    manifest=resolve(args.manifest);samples=load_manifest(manifest)
    wrapped,checkpoint=build_model(args.seed,device,args.checkpoint)
    # build_model moves only the detector; the physical bridge stays CPU FP32.
    if any(p.device.type!=device.type for p in wrapped.detector.parameters()):raise RuntimeError('Detector device mismatch.')
    if any(t.device.type!='cpu' for t in wrapped.bridge.buffers()):raise RuntimeError('Fusion bridge must stay on CPU.')
    outdir=resolve(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    report=dict(schema_version=1,method='state-contrast ISW-DETR',manifest_sha256=sha(manifest),
        seed=args.seed,visual_device=str(device),fusion_device='cpu',precision='float32',
        requested_batch_size=args.batch_size,sample_count=len(samples),input_order_preserved=True,
        tail_policy='Actual final sample count; no duplicate or dummy RGB padding.',
        event_padding='Variable E is zero-padded with false weight_mask; measured event_order is retained.',
        ground_truth_used=False,readout_alpha=.5,checkpoint_provenance=checkpoint,batches=[],outputs=[],
        protocol='PIL RGB; bilinear antialiased resize640x640; /255; visual FP32; CPU FP32 fusion; native top300 on visual device.',
        frozen_parameters=all(not p.requires_grad for p in wrapped.parameters()))
    checkpoint_short={k:v for k,v in checkpoint.items() if k!='build_log'}
    query_rows={key:[] for key in ('flat_visual_logits','visual_logits','isw_logits','boxes','candidate_mask','assigned_event','original_sizes','image_ids')}
    for batch_index,start in enumerate(range(0,len(samples),args.batch_size)):
        chunk=samples[start:start+args.batch_size]
        images=[]
        for row in chunk:
            with Image.open(row['image']) as image:images.append(image.convert('RGB'))
        x=torch.stack([image_tensor(image) for image in images]).to(device)
        modal=collate_modal([row['modal'] for row in chunk])
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();out=wrapped(x,modal_inputs=modal)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        if args.save_queries:
            for key,field in [('flat_visual_logits','pred_logits_visual'),('visual_logits','pred_logits_cfqd'),
                              ('isw_logits','pred_logits'),('boxes','pred_boxes'),
                              ('candidate_mask','state_candidate_mask'),('assigned_event','state_assigned_event')]:
                query_rows[key].append(out[field].detach().cpu().numpy())
            query_rows['original_sizes'].append(np.asarray([[image.width,image.height] for image in images],dtype=np.int64))
            query_rows['image_ids'].append(np.asarray([int(row['sample_id']) for row in chunk],dtype=np.int64))
        report['batches'].append(dict(batch_index=batch_index,input_indices=[r['index'] for r in chunk],
            actual_batch_size=len(chunk),forward_elapsed_s=elapsed))
        for i,(row,image) in enumerate(zip(chunk,images)):
            visual=out['pred_logits_cfqd'][i:i+1];final=out['pred_logits'][i:i+1]
            boxes=out['pred_boxes'][i:i+1];gate=bool(out['state_integrity_gate'][i]);code=int(out['state_integrity_rejection_code'][i])
            fallback=bool(torch.equal(final,visual)) if not gate else None
            if not gate and not fallback:raise RuntimeError('Rejected/missing transaction must return exact visual logits.')
            size=torch.tensor([[image.width,image.height]],device=device)
            payload=dict(schema_version=2,method=report['method'],seed=args.seed,sample_index=row['index'],
                sample_id=row['sample_id'],batch_index=batch_index,actual_batch_size=len(chunk),
                input_image_sha256=sha(row['image']),input_transaction_sha256=None if row['transaction'] is None else sha(row['transaction']),
                visual_device=str(device),fusion_device='cpu',precision='float32',readout_alpha=.5,
                ground_truth_used=False,all_parameters_frozen=report['frozen_parameters'],checkpoint_provenance=checkpoint_short,
                integrity_gate=gate,rejection_code=code,rejection_reasons=[k for k,v in wrapped.bridge.upstream.REJECTION_BITS.items() if code&v],
                exact_visual_fallback=fallback,scene_total_independent=False if row['record'] is None else bool(row['record'].get('scene_total_independent',False)),
                candidate_count=int(out['state_candidate_mask'][i].sum()),image_size=[image.width,image.height],
                detections=prediction_payload(final,boxes,size),visual_detections=prediction_payload(visual,boxes,size))
            if args.save_queries:
                payload['flat_visual_detections']=prediction_payload(out['pred_logits_visual'][i:i+1],boxes,size)
            filename=f"sample_{row['index']:06d}.json";write(outdir/filename,payload)
            report['outputs'].append(dict(sample_index=row['index'],sample_id=row['sample_id'],output=filename,
                integrity_gate=gate,exact_visual_fallback=fallback))
    if args.save_queries:
        np.savez_compressed(outdir/'queries.npz',**{key:np.concatenate(value) for key,value in query_rows.items()})
        report['queries_sha256']=sha(outdir/'queries.npz')
    write(outdir/'BATCH_PROVENANCE.json',report)
    print(json.dumps(dict(output_directory=str(outdir),samples=len(samples),
        actual_batch_sizes=[b['actual_batch_size'] for b in report['batches']],ground_truth_used=False)))

if __name__=='__main__':main()
