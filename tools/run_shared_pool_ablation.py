"""Replay visual/fixed 2x2 and both shared-pool auxiliary-head controls."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'reference_sources')]
from run_private_controls import infer,physical_cache,read,sha,tr
from shared_pool import prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inference-root','protocol-root','weights-root','output']:p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    mapping=torch.tensor(read(ROOT/'inference/inputs/taxonomy.json')['leaf_to_parent'])
    rows=[];baseline=[];hashes={}
    for seed in [42,3407,2026]:
        checkpoint=Path(args.weights_root)/f'detector_seed{seed}.pth'
        wrapped,_=infer.build_model(seed,torch.device('cpu'),str(checkpoint));hashes[str(checkpoint)]=sha(checkpoint)
        for split in ['internal','holdout']:
            query=Path(args.inference_root)/split/f'seed{seed}/predictions/queries.npz'
            arrays=dict(np.load(query));hashes[str(query)]=sha(query)
            manifest=read(Path(args.inference_root)/f'inputs/{split}/manifest.json')
            records=[read(row['transaction']) for row in manifest['samples']]
            data={m:physical_cache(arrays,records,wrapped.bridge.upstream,key) for m,key in [('flat','flat_visual_logits'),('identity','visual_logits')]}
            annotation=Path(args.protocol_root)/f'inputs/{split}.json';hashes[str(annotation)]=sha(annotation)
            for model in ['flat','identity']:
                for method in ['visual','fixed']:
                    logits=torch.as_tensor(data[model]['arrays']['logits']) if method=='visual' else prediction(data[model],data[model],mapping,'fixed')
                    baseline.append(dict(split=split,seed=seed,model=model,method=method,**tr.coco_metrics(data[model]['arrays'],logits,annotation)))
                for pool in ['flat','identity']:
                    logits=prediction(data[model],data[pool],mapping,'new')
                    poisoned={**data[model],'audits':None,'labels':float('nan'),'teacher_actions':float('nan')}
                    if not torch.equal(logits,prediction(poisoned,data[pool],mapping,'new')):raise ValueError('Label poison changed refinement')
                    for i,pack in enumerate(data[pool]['prepared']):
                        mask=torch.ones(300,dtype=torch.bool)
                        if pack is not None:mask[pack['candidates']]=False
                        if not torch.equal(logits[i,mask],torch.as_tensor(data[model]['arrays']['logits'][i])[mask]):raise ValueError('Noncandidate query changed')
                    rows.append(dict(split=split,seed=seed,model=model,reference_pool=pool,method='new',native_pool=model==pool,
                        all300=True,noncandidate_exact=True,label_poison_bitwise=True,**tr.coco_metrics(data[model]['arrays'],logits,annotation)))
            print(json.dumps(dict(split=split,seed=seed,evaluations=len(rows)+len(baseline))),flush=True)
    summary=[]
    for split in ['internal','holdout']:
        for condition in ['native','flat','identity']:
            select={m:[r for r in rows if r['split']==split and r['model']==m and (r['native_pool'] if condition=='native' else r['reference_pool']==condition)] for m in ['flat','identity']}
            gain=[100*(select['identity'][i]['ap']-select['flat'][i]['ap']) for i in range(3)]
            summary.append(dict(split=split,pool_condition=condition,flat_mean_ap_points=100*float(np.mean([r['ap'] for r in select['flat']])),
                identity_mean_ap_points=100*float(np.mean([r['ap'] for r in select['identity']])),conditional_IS_gain_mean_ap_points=float(np.mean(gain)),
                conditional_IS_gain_per_seed_ap_points=gain))
    report=dict(status='complete',rows=rows,baseline_rows=baseline,summary=summary,input_hashes=hashes,
                all48_evaluations_complete=len(rows)+len(baseline)==48,fresh_training=False)
    (out/'SHARED_POOL_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
