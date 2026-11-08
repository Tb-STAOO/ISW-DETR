"""Fit BQR/confidence controls or select the MAP coefficient on validation."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference_sources')]
import train_benefit_router as tr
from e1_core import ConfidencePredictionRouter,map_logits
from confidence_loss import confidence_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['bqr','confidence','map'],required=True)
    for name in ['cache-root','protocol-root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--seeds',nargs='+',type=int,choices=[42,3407,2026],default=[42,3407,2026])
    p.add_argument('--smoke',action='store_true',help='One optimization epoch only; not reported training')
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh training output directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    cache=Path(args.cache_root).resolve();protocol=Path(args.protocol_root).resolve()
    manifest=tr.read(cache/'ROUTER_CACHE_MANIFEST.json')
    if manifest['smoke_only'] and not args.smoke:p.error('A diagnostic cache cannot be used as full training')
    for r in manifest['rows']:
        if tr.sha(r['cache'])!=r['sha256']:raise ValueError('Prepared cache changed')
    sources=[Path(__file__),ROOT/'reference_sources/train_benefit_router.py',ROOT/'reference_sources/benefit_router.py']
    if args.method=='confidence':sources += [ROOT/'reference_sources/e1_core.py',ROOT/'reference_sources/confidence_loss.py']
    hashes={str(path):tr.sha(path) for path in sources+[cache/'ROUTER_CACHE_MANIFEST.json']}
    tr.write(out/'METHOD_INPUT_LOCK.json',dict(method=args.method,source_and_cache_sha256=hashes,smoke_only=args.smoke))
    if args.method=='map':
        rows=[];grid=[0,.125,.25,.5,1,1.5,2,3]
        for seed in args.seeds:
            data=tr.load_cache(cache/f'prepared/validation_seed{seed}.pt')
            for alpha in grid:
                logits,_=map_logits(data,alpha)
                rows.append(dict(seed=seed,alpha=alpha,**tr.coco_metrics(data['arrays'],logits,protocol/'inputs/validation.json')))
        means={alpha:float(np.mean([r['ap'] for r in rows if r['alpha']==alpha])) for alpha in grid}
        selected=max(grid,key=lambda a:(means[a],-a))
        tr.write(out/'map_selection.json',dict(alpha=selected,split='validation',grid=grid,rows=rows,seeds=args.seeds,
            smoke_only=args.smoke,input_sha256=hashes))
        print(json.dumps(dict(method='map',alpha=selected)),flush=True)
        return
    if args.method=='confidence':
        original=tr.protocol
        def protocol_for(a):
            value=original(a);value['method']='Confidence-prediction fusion control'
            value['loss'].pop('benefit_bce',None);value['loss']['confidence_mse']=.5;value.pop('benefit',None)
            value['confidence_targets']='Detached true-class softmax probabilities; true-background exp(-mean softplus)'
            value['routing']='g=2*pred_conf_C/(pred_conf_v+pred_conf_C)'
            return value
        tr.model_class=lambda:ConfidencePredictionRouter
        tr.loss_for=confidence_loss;tr.protocol=protocol_for
    reports=[]
    for seed in args.seeds:
        parameters=SimpleNamespace(seed=seed,mode='logit_noise',out=out,
            cache=cache/f'prepared/train_seed{seed}.pt',validation_cache=cache/f'prepared/validation_seed{seed}.pt',
            validation_annotations=protocol/'inputs/validation.json',epochs=1 if args.smoke else 30,
            batch_size=64,lr=.001,device='cpu',baseline_only=False,expected_baseline_ap=None)
        tr.train(parameters)
        source=out/f'logit_noise/seed{seed}/best.pth'
        payload=torch.load(source,map_location='cpu',weights_only=True)
        if args.method=='confidence':
            # Shared trainer metadata names BenefitRouter; correct this label
            # to the actual instantiated class, preserving every tensor.
            payload['class_name']='ConfidencePredictionRouter'
            torch.save(payload,source)
            for name in ['summary.json','SELECTION_LOCK.json']:
                path=source.parent/name
                summary=tr.read(path);summary['checkpoint_sha256']=tr.sha(source)
                tr.write(path,summary)
        destination=out/f'seed{seed}.pth';shutil.copy2(source,destination)
        reports.append(dict(seed=seed,class_name=payload['class_name'],checkpoint=str(destination),sha256=tr.sha(destination),
            selection=payload['selection']))
    tr.write(out/'ROUTER_TRAINING_RESULT.json',dict(status='complete',method=args.method,smoke_only=args.smoke,
        seeds=args.seeds,epoch_budget=1 if args.smoke else 30,rows=reports,input_sha256=hashes))


if __name__=='__main__':main()
