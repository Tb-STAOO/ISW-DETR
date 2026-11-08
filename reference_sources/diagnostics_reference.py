"""Functions preserved from the original experiment implementation."""
from pathlib import Path
import sys, contextlib, io, collections
import numpy as np

def fixed_prediction_diagnostics():
    """Re-evaluate per-class AP and ownership on frozen native query outputs."""
    import torch
    from scipy.optimize import linear_sum_assignment
    sys.path.insert(0,str(ROOT/'RT_DETR/tools/ambiguity_fusion_20261013'))
    import train_benefit_router as tr
    sys.path.insert(0,str(ROOT/'RT_DETR/tools/identity_state_decomposition_20261021'))
    from transforms import transform
    sys.path.insert(0,str(tr.NATIVE))
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(2)
    taxonomy=ROOT/'inference/inputs/taxonomy.json'
    mapping=np.asarray(read(taxonomy)['leaf_to_parent'])
    perclass=[];owners=[];corrections=[];cases=[]
    post=RTDETRPostProcessor(num_classes=28,num_top_queries=300,remap_mscoco_category=False).eval()
    methods=('original_fixed_CVSR','original_BQR',MAIN)
    for split in ('internal','holdout'):
        ann=read(PACKAGE/f'inputs/{split}.json')
        imageinfo={im['id']:im for im in ann['images']}
        for seed in SEEDS:
            path=tr.OLD_PREPARED/f'{split}_seed{seed}.pt';SOURCE[str(path)]=sha(path)
            data=tr.load_cache(path);pack=tr.pack(data)
            path=R21/f'{split}/seed{seed}/predictions.npz';SOURCE[str(path)]=sha(path)
            saved=np.load(path);assert np.array_equal(saved['image_ids'],data['arrays']['image_ids'])
            pred={m:saved[m] for m in methods}
            # Exact native class-aware top-300 output and standard COCO evaluator.
            for m in methods:
                records=[]
                for start in range(0,len(data['prepared']),64):
                    res=post(dict(pred_logits=torch.from_numpy(pred[m][start:start+64]),
                        pred_boxes=torch.as_tensor(data['arrays']['boxes'][start:start+64])),
                        torch.as_tensor(data['arrays']['original_sizes'][start:start+64]))
                    for iid,r in zip(data['arrays']['image_ids'][start:start+64],res):
                        for b,s,l in zip(r['boxes'].tolist(),r['scores'].tolist(),r['labels'].tolist()):
                            x1,y1,x2,y2=b
                            records.append(dict(image_id=int(iid),category_id=int(l),bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)],score=float(s)))
                with contextlib.redirect_stdout(io.StringIO()):
                    gt=COCO();gt.dataset=ann;gt.createIndex();dt=gt.loadRes(records)
                    ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=sorted(map(int,data['arrays']['image_ids']))
                    ev.evaluate();ev.accumulate();ev.summarize()
                expected=next(r['ap'] for r in read(R21/'PRIVATE_CONTROLS.json')['rows'] if r['split']==split and r['seed']==seed and r['method']==m)
                assert abs(float(ev.stats[0])-expected)<1e-12,(split,seed,m,ev.stats[0],expected)
                precision=ev.eval['precision'][:,:,:,0,2]
                aps=[]
                for c in range(28):
                    vals=precision[:,:,c];aps.append(float(vals[vals>=0].mean()*100))
                perclass.append(dict(split=split,seed=seed,method=m,ap_points=100*float(ev.stats[0]),class_ap_points=aps))
                print('PERCLASS',split,seed,m,flush=True)
            for name,mode in [('original','original'),('mean','mean_erase'),('max','max_erase')]:
                good=total=observable=equivalent=all_edges=0
                for i,(p,audit) in enumerate(zip(data['prepared'],data['audits'])):
                    if p is None:continue
                    k=len(p['z']);e=len(p['logp']);z=pack['z'][i,:k]
                    z=transform(z,torch.as_tensor(mapping),mode)
                    hard,_,_=tr.predicted_assignment(z,pack['scene'][i],pack['event_residual'][i,:e])
                    predicted=hard.argmax(0).numpy();own=np.asarray(audit['owners']);known=own>=0
                    assert len(own)==e
                    all_edges+=e
                    total+=int(known.sum());good+=int(((predicted==own)&known).sum())
                    labels=np.asarray(audit['labels']);fg=labels[predicted]>=0
                    observable+=int((known&fg).sum())
                    for j in np.where(known&fg)[0]:equivalent+=int(labels[predicted[j]]==labels[own[j]])
                owners.append(dict(split=split,seed=seed,association=name,all_edges=all_edges,known_edges=total,exact_owner_edges=good,
                    exact_owner_percent=100*good/all_edges,observable_owner_percent=100*good/total,
                    assigned_foreground_edges=observable,class_equivalent_edges=equivalent))
            count=collections.Counter()
            for i,(p,audit) in enumerate(zip(data['prepared'],data['audits'])):
                if p is None:continue
                ids=np.asarray(p['candidates']);truth=np.asarray(audit['labels']);fg=truth>=0
                old=pred['original_BQR'][i,ids].argmax(-1);new=pred[MAIN][i,ids].argmax(-1)
                oc=(old==truth)&fg;nc=(new==truth)&fg;corrected=~oc&nc;harmed=oc&~nc
                op=mapping[old];np_=mapping[new];yp=np.where(fg,mapping[np.maximum(truth,0)],-1)
                state=fg&(op==yp)&(np_==yp)
                count.update(foreground=int(fg.sum()),corrected=int(corrected.sum()),harmed=int(harmed.sum()),
                    state_corrected=int((state&corrected).sum()),state_harmed=int((state&harmed).sum()),
                    identity_corrected=int((corrected&~state).sum()),identity_harmed=int((harmed&~state).sum()))
                if split=='holdout' and seed==42:
                    for kind,mask in [('corrected',corrected),('harmed',harmed)]:
                        for j in np.where(mask)[0]:
                            iid=int(data['arrays']['image_ids'][i]);box=np.asarray(data['arrays']['boxes'][i,ids[j]]).tolist()
                            cases.append(dict(kind=kind,image_id=iid,query_id=int(ids[j]),candidate_index=int(j),
                                before=int(old[j]),after=int(new[j]),truth=int(truth[j]),box_cxcywh=box,
                                file_name=imageinfo[iid]['file_name']))
            corrections.append(dict(split=split,seed=seed,**count))
    selected=[]
    for kind in ('corrected','harmed'):
        group=sorted((c for c in cases if c['kind']==kind),key=lambda c:(c['image_id'],c['query_id']))
        selected.append(group[0])
    obj=dict(per_class=perclass,ownership=owners,corrections=corrections,selected_cases=selected,
        case_selection='seed42 measured split; first image ID then first query ID separately for corrected and harmed; fixed original geometry',
        ownership_definition='Exact predicted Hungarian owner versus cached geometric owner / all event edges, including unobservable owner failures; observable-only percentage also retained. Not semantic object identity.',
        class_definition='Native top300 COCO class AP; fixed-candidate argmax correction counts are a separate diagnostic.',
        no_training=True,no_new_inference=True)
    write(DATA/'PREDICTION_DIAGNOSTICS.json',obj)
    append_diagnostic_values(obj)
    return obj

