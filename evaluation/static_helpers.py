"""Exploratory FPB-derived food x native-portion pilot; shared RGB outputs.

No physical transaction gate is simulated. Numeric eligibility means only that
the static public-data record list is finite and positive. Test outcomes are not
used to select a dataset, noise value, scoring coefficient or class grouping.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RUNTIME = ROOT/'runtime'
REFINEMENT_PATH = ROOT/'inference/adapters/refinement.py'
sys.path.insert(0, str(RUNTIME))
from src.core import YAMLConfig
from src.zoo.rtdetr.cvsr import ConservationVerifiedSetRefinement
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

spec = importlib.util.spec_from_file_location('external_state_refinement', REFINEMENT_PATH)
refinement_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement_module)
StateNeutralRefinement = refinement_module.StateNeutralRefinement

METHODS = ('rgb', 'gaussian_broadcast', 'original_original', 'original_half',
           'mean_original', 'mean_full', 'mean_half')
NAMES = ('AP','AP50','AP75','AP_small','AP_medium','AP_large',
         'AR1','AR10','AR100','AR_small','AR_medium','AR_large')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_train_priors(data_root, classes):
    """Owner labels enter train-only fitting, never inference/association."""
    path = data_root / 'evaluation_sidecars/train_owners.json'
    source = read(path)
    values = [[] for _ in range(classes)]
    for image in source:
        if image.get('split','train') != 'train':
            raise ValueError('Non-train owner sidecar passed to prior fit.')
        for owner in image['owners_by_record']:
            label, mass = int(owner['category_id']), float(owner['mass_g'])
            if not 0 <= label < classes or not np.isfinite(mass) or mass <= 0:
                raise ValueError('Invalid training mass/label.')
            values[label].append(mass)
    if any(not masses for masses in values):
        raise ValueError('Every class must have training mass observations.')
    priors = [dict(category_id=label, count=len(masses), mean_g=float(np.mean(masses)),
                   std_g=max(float(np.std(masses,ddof=1)) if len(masses)>1 else 0.,1.))
              for label,masses in enumerate(values)]
    return priors, path


def taxonomy_from_categories(categories):
    """Explicit source taxonomy, never inferred from labels or mass thresholds."""
    rows=sorted(categories,key=lambda row:int(row['id']))
    if [int(row['id']) for row in rows]!=list(range(len(rows))):
        raise ValueError('Leaf class ids must be contiguous and start from zero.')
    mapping=torch.tensor([int(row['identity_id']) for row in rows],dtype=torch.long)
    if (mapping<0).any():
        raise ValueError('Negative identity id.')
    for identity in mapping.unique().tolist():
        members=[row for row in rows if int(row['identity_id'])==identity]
        if sorted(int(row['state_id']) for row in members)!=[0,1,2]:
            raise ValueError('Each FPB identity requires native small/average/big states 0,1,2.')
    return mapping


def make_evidence(masses, priors, sigma=.25):
    """Native CVSR likelihood/evidence equations under declared static sigma."""
    masses = torch.as_tensor(masses,dtype=torch.float32)
    mu = torch.tensor([row['mean_g'] for row in priors],dtype=torch.float32)
    std = torch.tensor([row['std_g'] for row in priors],dtype=torch.float32)
    if masses.ndim!=1 or not masses.numel() or not torch.isfinite(masses).all() or (masses<=0).any():
        return None
    total_std = (std.square()+float(sigma)**2).sqrt()
    log_likelihood = -.5*((masses[:,None]-mu[None])/total_std[None]).square()-total_std.log()[None]
    logp = torch.log_softmax(log_likelihood,dim=-1)
    probability = logp.exp().max(dim=0).values
    log_scene = probability.clamp_min(1e-8).log()
    scene = (log_scene+np.log(len(priors))).clamp(-6,6)
    residual = (logp-log_scene[None]).clamp(-6,0)
    if not torch.isfinite(scene).all() or not torch.isfinite(residual).all():
        return None
    return scene,residual,logp


def parent_mean(logits,mapping):
    result=logits.clone()
    for parent in mapping.unique(sorted=True):
        ids=(mapping==parent).nonzero(as_tuple=False).flatten()
        result[:,ids]=logits[:,ids].mean(-1,keepdim=True)
    return result


def original_association(logits,ids,scene,residual,mapping,alpha):
    """Same rectangular Hungarian; original logits in association utility."""
    selected=logits[ids]
    utility=torch.logsumexp((selected+2*scene)[:,None,:]+residual[None,:,:],-1)
    rows,cols=linear_sum_assignment(-utility.numpy())
    proposal=torch.zeros_like(selected)
    proposal[torch.as_tensor(rows)]=residual[torch.as_tensor(cols)]
    refined=logits.clone()
    refined[ids]=selected-alpha*(selected-parent_mean(selected,mapping))+(2*scene+.2*proposal)
    assignment=torch.full((logits.shape[0],),-1,dtype=torch.long)
    assignment[ids[torch.as_tensor(rows)]]=torch.as_tensor(cols)
    return refined,assignment


def score_one(logits,boxes,masses,priors,mapping,sigma=.25):
    output={name:logits.clone() for name in METHODS}
    evidence=make_evidence(masses,priors,sigma)
    if evidence is None:
        return output,dict(eligible=False,candidate_count=0,event_count=len(masses)),{}
    scene,residual,_=evidence
    # Reuse native candidate helper directly, without constructing a telemetry
    # module or declaring a synthetic transaction to have passed its gate.
    ids=ConservationVerifiedSetRefinement._select_candidates(
        SimpleNamespace(candidate_iou_threshold=.9),logits,boxes,min(len(logits),len(masses)+12))
    selected=torch.zeros(len(logits),dtype=torch.bool)
    selected[ids]=True
    output['gaussian_broadcast'][ids]=logits[ids]+2*scene
    assignments={}
    for name,alpha in [('original_original',0.),('original_half',.5)]:
        output[name],assignments[name]=original_association(logits,ids,scene,residual,mapping,alpha)
    for name,alpha in [('mean_original',0.),('mean_full',1.),('mean_half',.5)]:
        module=StateNeutralRefinement(mapping,alpha=alpha)
        result=module(logits[None],ids[None],torch.ones(1,len(ids),dtype=torch.bool),
                      scene[None],residual[None],torch.ones(1,len(masses),dtype=torch.bool),
                      # Refiner accepts upstream usability. Here it is STATIC
                      # numeric eligibility, never a physical telemetry verdict.
                      torch.ones(1,dtype=torch.bool))
        output[name]=result['pred_logits'][0]
        assignments[name]=result['assigned_event'][0]
    for name,refined in output.items():
        if not torch.equal(refined[~selected],logits[~selected]):
            raise AssertionError(f'Noncandidate logits changed: {name}')
        if not torch.isfinite(refined).all():
            raise AssertionError(f'Nonfinite logits: {name}')
    return output,dict(eligible=True,candidate_count=len(ids),event_count=len(masses),
                       candidate_indices=ids.tolist(),noncandidate_passthrough=True),assignments


def image_batches(images,data_root,size,batch_size):
    for start in range(0,len(images),batch_size):
        rows=images[start:start+batch_size]
        tensors=[]
        for row in rows:
            with Image.open(data_root/'images'/row['file_name']) as handle:
                image=handle.convert('RGB')
                if image.size!=(row['width'],row['height']):
                    raise ValueError(f'COCO image-size mismatch: {row["id"]}')
                image=TF.resize(image,size,antialias=True)
                tensors.append(TF.pil_to_tensor(image).float()/255.)
        yield rows,torch.stack(tensors)


def export_raw(cfg,checkpoint,images,data_root,device,batch_size,out_path):
    model=cfg.model
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ema='ema' in source and isinstance(source['ema'],dict) and 'module' in source['ema']
    weights=source['ema']['module'] if ema else source['model']
    model.load_state_dict(weights,strict=True)
    model=model.to(device).eval()
    size=list(cfg.yaml_cfg['eval_spatial_size'])
    classes=int(cfg.yaml_cfg['num_classes'])
    all_logits,all_boxes=[],[]
    started=time.perf_counter()
    with torch.inference_mode():
        for rows,images_tensor in image_batches(images,data_root,size,batch_size):
            result=model(images_tensor.to(device))
            logits=result['pred_logits'].detach().float().cpu().numpy()
            boxes=result['pred_boxes'].detach().float().cpu().numpy()
            if logits.shape[1:]!=(300,classes) or boxes.shape[1:]!=(300,4):
                raise ValueError('Unexpected raw output dimensions.')
            all_logits.append(logits)
            all_boxes.append(boxes)
            print(f'Raw RGB export {sum(map(len,all_logits))}/{len(images)}',flush=True)
    logits=np.concatenate(all_logits)
    boxes=np.concatenate(all_boxes)
    np.savez_compressed(out_path,logits=logits,boxes=boxes,
                        image_ids=np.array([int(row['id']) for row in images],dtype=np.int64),
                        orig_sizes=np.array([[row['width'],row['height']] for row in images],dtype=np.int64))
    return dict(checkpoint_state='ema.module' if ema else 'model',
                export_seconds=time.perf_counter()-started,precision='FP32 inference, no autocast')


def as_coco(postprocessor,logits,boxes,image_id,size):
    output=postprocessor({'pred_logits':logits[None],'pred_boxes':boxes[None]},
                         torch.as_tensor([size],dtype=torch.int64))[0]
    results=[]
    for label,box,score in zip(output['labels'],output['boxes'],output['scores']):
        box=box.tolist()
        box[2]-=box[0]
        box[3]-=box[1]
        results.append(dict(image_id=int(image_id),category_id=int(label),bbox=box,score=float(score)))
    return results


def coco_metrics(gt_path,predictions,image_ids,stdout_path):
    stream=io.StringIO()
    with redirect_stdout(stream):
        gt=COCO(str(gt_path))
        dt=gt.loadRes(predictions)
        evaluator=COCOeval(gt,dt,'bbox')
        evaluator.params.imgIds=list(map(int,image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    Path(stdout_path).write_text(stream.getvalue(),encoding='utf-8')
    return {name:float(value)*100 if value>=0 else None for name,value in zip(NAMES,evaluator.stats)}


def single_target_metrics(predictions,annotation,mapping):
    """Post-hoc GT evaluation only: global highest-scoring query-class pair."""
    targets={}
    for row in annotation['annotations']:
        if int(row.get('iscrowd',0)):
            continue
        image_id=int(row['image_id'])
        if image_id in targets:
            raise ValueError('FPB portion component protocol requires exactly one target per image.')
        targets[image_id]=row
    top={}
    for row in predictions:
        image_id=int(row['image_id'])
        if image_id not in top or row['score']>top[image_id]['score']:
            top[image_id]=row
    correct_label=correct_identity=correct_joint=0
    ious=[]
    for image_id,truth in targets.items():
        predicted=top[image_id]
        predicted_label,true_label=int(predicted['category_id']),int(truth['category_id'])
        is_label=predicted_label==true_label
        correct_label+=is_label
        correct_identity+=int(mapping[predicted_label])==int(mapping[true_label])
        a,b=predicted['bbox'],truth['bbox']
        left,upper=max(a[0],b[0]),max(a[1],b[1])
        right,lower=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
        intersection=max(0.,right-left)*max(0.,lower-upper)
        union=max(0.,a[2])*max(0.,a[3])+max(0.,b[2])*max(0.,b[3])-intersection
        overlap=intersection/union if union>0 else 0.
        ious.append(overlap)
        correct_joint+=is_label and overlap>=.5
    count=len(targets)
    if count!=len(annotation['images']) or count!=len(top):
        raise ValueError('Single-target metric image inventory mismatch.')
    return dict(top1_label_accuracy_pct=100.*correct_label/count,
                top1_identity_accuracy_pct=100.*correct_identity/count,
                top1_label_and_iou50_accuracy_pct=100.*correct_joint/count,
                top1_mean_iou=float(np.mean(ious)),top1_count=count)


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    data_root,output=Path(args.data_root),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.split=='test' and (output/'test_summary.json').exists():
        raise RuntimeError('Completed test results already exist; preserve the locked one-pass evaluation.')
    annotation_path=data_root/f'annotations/{args.split}.json'
    observation_path=data_root/f'observations/{args.split}.json'
    annotation=read(annotation_path)
    images=sorted(annotation['images'],key=lambda row:int(row['id']))
    if not images:
        raise ValueError('The frozen selected task has zero official dev photos; skip evaluation, use fixed final-epoch EMA on official test once.')
    mapping=taxonomy_from_categories(annotation['categories'])
    classes=len(mapping)
    train_annotation_path=data_root/'annotations/train.json'
    train_categories=read(train_annotation_path)['categories']
    train_mapping=taxonomy_from_categories(train_categories)
    if [row for row in sorted(annotation['categories'],key=lambda r:int(r['id']))]!=[
            row for row in sorted(train_categories,key=lambda r:int(r['id']))]:
        raise ValueError('Evaluation categories and frozen training categories differ.')
    if classes!=24 or not torch.equal(mapping,train_mapping):
        raise ValueError('Expected the frozen 24-class, 8-food x 3-native-portion task.')
    obs={int(row['image_id']):row for row in read(observation_path)}
    if len(obs)!=len(images) or set(obs)!={int(row['id']) for row in images}:
        raise ValueError('Observation and evaluation image sets differ.')
    for row in obs.values():
        if row.get('mass_unit')!='g':
            raise ValueError('Static protocol expects grams, with unit conversion in data adapter.')
        if len(row['records'])!=1:
            raise ValueError('FPB portion pilot requires exactly one static mass per photo.')
    cfg=YAMLConfig(args.config,PResNet={'pretrained':False})
    if int(cfg.yaml_cfg['num_classes'])!=classes:
        raise ValueError('Training configuration and dataset class counts differ.')
    priors,prior_path=fit_train_priors(data_root,classes)
    write(output/'train_only_mass_priors.json',dict(class_delta_gaussians=priors,
          training_owner_sidecar=str(prior_path),sha256=sha(prior_path),
          standard_deviation='sample SD ddof=1, floor 1 g',observation_sigma_g=.25))
    if args.split=='test':
        protocol_lock=dict(checkpoint_sha256=sha(args.checkpoint),config_sha256=sha(args.config),
                           evaluator_sha256=sha(__file__),training_prior_source_sha256=sha(prior_path),
                           leaf_to_parent=mapping.tolist(),alpha=.5,beta=2.,eta=.2,
                           observation_sigma_g=.25,class_std_floor_g=1.,
                           created_before_test_inference_utc=datetime.now(timezone.utc).isoformat())
        lock_path=output/'test_protocol_lock.json'
        if lock_path.exists():
            previous=read(lock_path)
            if any(previous.get(key)!=value for key,value in protocol_lock.items()
                   if key!='created_before_test_inference_utc'):
                raise RuntimeError('Attempted changed method/config/checkpoint after locking test protocol.')
        else:
            write(lock_path,protocol_lock)
    raw_path=Path(args.raw_cache) if args.raw_cache else output/f'{args.split}_raw_rgb.npz'
    raw_meta_path=Path(str(raw_path)+'.metadata.json')
    required_raw_provenance=dict(checkpoint_sha256=sha(args.checkpoint),
                                 config_sha256=sha(args.config),annotation_sha256=sha(annotation_path))
    if args.raw_cache:
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_provenance=read(raw_meta_path)
        if any(raw_provenance.get(key)!=value for key,value in required_raw_provenance.items()):
            raise ValueError('Raw-cache checkpoint/config/annotation provenance mismatch.')
        if raw_provenance.get('raw_cache_sha256')!=sha(raw_path):
            raise ValueError('Raw-cache digest mismatch.')
        export_meta=dict(cache_reused=True)
    else:
        export_meta=export_raw(cfg,args.checkpoint,images,data_root,args.device,args.batch_size,raw_path)
        write(raw_meta_path,dict(**required_raw_provenance,raw_cache_sha256=sha(raw_path),**export_meta))
    with np.load(raw_path,allow_pickle=False) as cache:
        logits=torch.from_numpy(cache['logits'].copy()).float()
        boxes=torch.from_numpy(cache['boxes'].copy()).float()
        image_ids=cache['image_ids'].copy()
        sizes=cache['orig_sizes'].copy()
    if list(image_ids)!=[int(row['id']) for row in images]:
        raise ValueError('Raw-cache image ids/order mismatch.')
    if not np.array_equal(sizes,np.array([[row['width'],row['height']] for row in images])):
        raise ValueError('Raw-cache image dimensions mismatch.')
    if tuple(logits.shape)!=(len(images),300,classes) or tuple(boxes.shape)!=(len(images),300,4):
        raise ValueError('Raw-cache dimensions mismatch.')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise ValueError('Invalid cached visual output.')
    # Explicit native portion taxonomy comes from frozen category metadata.
    # Neither owner labels nor native portion labels enter image inference.
    pred={method:[] for method in METHODS}
    diagnostics=[]
    refined_cache={method:[] for method in METHODS}
    postprocessor=cfg.postprocessor.cpu().eval()
    with torch.inference_mode():
        for idx,image_id in enumerate(image_ids):
            records=obs[int(image_id)]['records']
            masses=[float(row['mass_g']) for row in records]
            if len({int(row['record_id']) for row in records})!=len(records):
                raise ValueError('Duplicate static record ids.')
            values,diagnostic,assignments=score_one(logits[idx],boxes[idx],masses,priors,mapping)
            diagnostic.update(image_id=int(image_id),scene_id=obs[int(image_id)].get('scene_id'),
                              physical_transaction_gate_evaluated=False,
                              static_record_ids=[int(row['record_id']) for row in records])
            diagnostics.append(diagnostic)
            for method in METHODS:
                refined_cache[method].append(values[method].numpy())
                pred[method].extend(as_coco(postprocessor,values[method],boxes[idx],image_id,sizes[idx]))
    metrics={}
    for method in METHODS:
        write(output/f'{args.split}_{method}_predictions.json',pred[method])
        metrics[method]=coco_metrics(annotation_path,pred[method],image_ids,
                                     output/f'{args.split}_{method}_coco_eval.txt')
        metrics[method].update(single_target_metrics(pred[method],annotation,mapping))
        print(method,metrics[method],flush=True)
    np.savez_compressed(output/f'{args.split}_all_refined_logits.npz',
                        **{method:np.stack(rows) for method,rows in refined_cache.items()})
    baseline=metrics['rgb']['AP']
    fixed=metrics['original_original']['AP']
    for row in metrics.values():
        row['delta_AP_vs_rgb_pp']=row['AP']-baseline
        row['delta_AP_vs_fixed_pp']=row['AP']-fixed
    write(output/f'{args.split}_numeric_eligibility_and_candidates.json',diagnostics)
    summary=dict(
        status='exploratory adaptation pilot',created_utc=datetime.now(timezone.utc).isoformat(),
        training_seed=42,training_seed_source='pilot run configuration; single seed, no CI',
        split=args.split,images=len(images),metrics=metrics,
        observed_leaf_classes=sorted({int(row['category_id']) for row in annotation['annotations']}),
        observed_identity_ids=sorted({int(mapping[int(row['category_id'])]) for row in annotation['annotations']}),
        protocol=dict(name='public static mass-annotation component protocol',
                      real_transaction_gate_tested=False,numeric_eligible=sum(row['eligible'] for row in diagnostics),
                      caveat='eligibility is not a physical sensor conservation verdict',
                      record_count_source='one annotated food target and one measured mass per selected photo; same for all mass methods',
                      state_source='original filename portion (small/average/big), not a mass-threshold-derived state',
                      sampling_unit='official photo split; no assertion of independent physical portion instances',
                      scope='closed-catalog FPB-derived single-food portion task; no multi-record ownership claim',
                      checkpoint_selection='fixed final epoch 20 EMA module; no dev/test checkpoint selection; seven test branches once after lock',
                      leaf_to_parent=mapping.tolist(),observation_sigma_g=.25,class_std_floor_g=1.,
                      class_stats_fit_split='train',candidate_extra=12,candidate_iou_threshold=.9,
                      scene_beta=2.,residual_strength=.2,evidence_clip=6.,
                      gaussian_broadcast='z+2*scene on the shared candidate set; no record assignment',
                      mean_full='mean association and alpha=1, i.e. no original state contrast retained',
                      mean_original='mean association and alpha=0',mean_half='mean association and alpha=.5',
                      COCO_AP='canonical IoU .50:.95; maxDets [1,10,100]; 300 raw detections exported',
                      official_test_split=args.split=='test',
                      physical_instance_independence_verified=False,confidence_intervals=False),
        qa=dict(shared_raw_logits=True,shared_raw_boxes=True,boxes_unchanged=True,
                shared_candidates=True,noncandidate_exact_passthrough=True,
                no_ground_truth_owners_in_inference=True),
        sources={str(path):sha(path) for path in (Path(args.config),Path(args.checkpoint),annotation_path,
                 observation_path,prior_path,train_annotation_path,REFINEMENT_PATH,Path(__file__),raw_path,raw_meta_path,
                 RUNTIME/'src/zoo/rtdetr/cvsr.py',RUNTIME/'src/zoo/rtdetr/rtdetr_postprocessor.py')},
        raw_cache=str(raw_path),export=export_meta)
    write(output/f'{args.split}_summary.json',summary)
    table=['# FPB-derived native-portion static-mass pilot','',
           'Exploratory single-seed adaptation using the official photograph split; no physical transaction or physical-instance-independence claim.','',
           '| Method | AP | AP50 | AP75 | Top1 label % | Top1 label + IoU≥.5 % | ΔAP vs RGB | ΔAP vs fixed |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in metrics.items():
        def number(key):
            return 'n/a' if row[key] is None else f'{row[key]:.4f}'
        table.append('| '+' | '.join([name]+[number(key) for key in ('AP','AP50','AP75','top1_label_accuracy_pct','top1_label_and_iou50_accuracy_pct',
                       'delta_AP_vs_rgb_pp','delta_AP_vs_fixed_pp')])+' |')
    (output/f'{args.split}_RESULTS.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
