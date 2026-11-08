"""C-CCM full visual-branch training with controlled static-mass fusion.

Each flat or seven-state I-S checkpoint uses the same locked seven scoring
methods. Static configuration masses do not validate physical transactions.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,gc,hashlib,importlib.util,json,sys,time
import numpy as np
import torch
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
COMMON=Path(__file__).resolve().with_name('static_helpers.py')
spec=importlib.util.spec_from_file_location('locked_shared_static_score_helpers',COMMON)
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
read,write,sha=shared.read,shared.write,shared.sha
METHODS=shared.METHODS

def taxonomy(categories):
    rows=sorted(categories,key=lambda row:int(row['id']))
    if [int(row['id']) for row in rows]!=list(range(len(rows))):raise ValueError('Non-contiguous native leaf IDs')
    mapping=torch.tensor([int(row['identity_id']) for row in rows])
    if not len(rows) or sorted(mapping.unique().tolist())!=list(range(len(mapping.unique()))):
        raise ValueError('Native container parent IDs must be contiguous from zero')
    return mapping

def owner_entries(photo):
    entries=photo.get('owners',photo.get('owners_by_record'))
    if not isinstance(entries,list):raise ValueError('Missing separate owner list')
    return entries

def source_provenance(protocol,protocol_path):
    """Validate locked author metadata/conversion without model inputs.

    The selection digest is canonical JSON; source/output digests are file-byte
    SHA256. Source labels checked here never become RGB/fusion input features.
    """
    source=Path(protocol['source_annotations']).parent
    native_path=Path(protocol['source_annotations']);mass_path=Path(protocol['mass_source_annotations'])
    conversion=Path(protocol['source_conversion_file'])
    paths=[Path(protocol_path),native_path,mass_path,conversion,
           Path(protocol['selection_file']),Path(protocol['mass_lookup_table'])]
    for name,expected in protocol['source_hashes'].items():
        path=source/name
        if not path.is_file() or sha(path)!=expected:raise ValueError(f'Locked author/source metadata changed: {path}')
        paths.append(path)
    if sha(conversion)!=protocol['source_conversion_sha256']:
        raise ValueError('Conversion source changed after source-protocol lock')
    selection=read(protocol['selection_file'])
    canonical=hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if canonical!=protocol['selection_sha256']:raise ValueError('Canonical frozen selection changed')
    counts={name:sum(row['split']==name for row in selection) for name in protocol['counts']}
    if counts!=protocol['counts']:raise ValueError('Frozen selection split counts changed')
    native={int(row['id']):row for row in read(native_path)['annotations']}
    cats={int(row['id']):row for row in protocol['categories']}
    for row in selection:
        original=native[int(row['image_id'])];category=cats[int(row['category_id'])]
        if (int(original['containerID']),int(original['filling_type']),int(original['filling_level']))!=(
                int(category['native_container_id']),int(category['native_filling_type']),int(category['native_filling_level'])):
            raise ValueError('Frozen selection label differs from original C-CCM attributes')
        if int(original['view'])!=int(protocol['split_views'][row['split']]):
            raise ValueError('Frozen selection crossed the locked native-view split')
    mass_rows={int(row['id']):row for row in read(mass_path)['annotations']}
    table=read(protocol['mass_lookup_table'])
    if sorted(int(row['category_id']) for row in table)!=list(range(protocol['num_classes'])):
        raise ValueError('Configuration mass table has a different class inventory')
    for row in table:
        c=cats[int(row['category_id'])]
        if len(row['source_annotation_ids'])!=protocol['mass_source_repeats_per_configuration']:
            raise ValueError('Configuration mass source support changed')
        for identifier in row['source_annotation_ids']:
            original=mass_rows[int(identifier)]
            if (int(original['container id']),int(original['filling type']),int(original['filling level']))!=(
                    int(c['native_container_id']),int(c['native_filling_type']),int(c['native_filling_level'])):
                raise ValueError('Configuration mass-table source condition changed')
            for source_key,table_key in [('container mass','container_mass_g'),
                    ('filling mass','filling_mass_g'),('object mass','object_mass_g')]:
                if abs(float(original[source_key])-float(row[table_key]))>1e-8:
                    raise ValueError('Configuration mass differs from original CCM annotation')
        if float(row['object_mass_g'])<=0 or abs(float(row['object_mass_g'])-
                float(row['container_mass_g'])-float(row['filling_mass_g']))>1e-8:
            raise ValueError('Configuration object mass is not positive container plus content mass')
    paths=list(dict.fromkeys(paths))
    return paths,dict(closed_source_chain=True,source_protocol_sha256=sha(protocol_path),
        source_conversion_sha256=sha(conversion),selection_canonical_sha256=canonical,
        selection_file_sha256=sha(protocol['selection_file']),mass_table_sha256=sha(protocol['mass_lookup_table']),
        original_C_CCM_and_CCM_mass_annotations_verified=True,
        known_configuration_mass_values_verified=len(table),selected_photo_counts=counts)

def train_priors(root,classes):
    path=root/'evaluation_sidecars/train_owners.json';values=[[] for _ in range(classes)]
    for photo in read(path):
        if photo['split']!='train':raise ValueError('Prior data crossed training split')
        for owner in owner_entries(photo):
            label,mass=int(owner['category_id']),float(owner['mass_g'])
            if not 0<=label<classes or not np.isfinite(mass) or mass<=0:raise ValueError('Invalid train mass')
            values[label].append(mass)
    if any(not row for row in values):raise ValueError('Missing training mass for a native leaf ID')
    return [dict(category_id=i,count=len(v),mean_g=float(np.mean(v)),std_g=max(float(np.std(v,ddof=1)) if len(v)>1 else 0.,1.))
            for i,v in enumerate(values)],path

def export(cfg,checkpoint,images,root,device,batch_size,out):
    model=cfg.model
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    use_ema='ema' in source and 'module' in source['ema']
    model.load_state_dict(source['ema']['module'] if use_ema else source['model'],strict=True)
    model=model.to(device).eval();size=cfg.yaml_cfg['eval_spatial_size']
    logits=[];boxes=[];started=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(images),batch_size):
            batch=images[offset:offset+batch_size];inputs=[]
            for photo in batch:
                with Image.open(root/photo['file_name']) as image:
                    image=image.convert('RGB').resize((int(size[1]),int(size[0])),Image.Resampling.BILINEAR)
                    inputs.append(shared.TF.to_tensor(image))
            result=model(torch.stack(inputs).to(device))
            logits.append(result['pred_logits'].detach().float().cpu().numpy())
            boxes.append(result['pred_boxes'].detach().float().cpu().numpy())
    np.savez_compressed(out,logits=np.concatenate(logits),boxes=np.concatenate(boxes),
                        image_ids=np.array([int(x['id']) for x in images]),
                        orig_sizes=np.array([[x['width'],x['height']] for x in images]))
    return dict(checkpoint_state='ema.module' if use_ema else 'model',precision='FP32 inference',seconds=time.perf_counter()-started)

def geometry(logits,boxes,image,truth,event_count):
    if not truth:return {}
    ids=shared.ConservationVerifiedSetRefinement._select_candidates(
        shared.SimpleNamespace(candidate_iou_threshold=.9),logits,boxes,min(len(logits),event_count+12))
    size=np.array([image['width'],image['height']]);b=boxes[ids].numpy()
    left=(b[:,:2]-.5*b[:,2:])*size;right=(b[:,:2]+.5*b[:,2:])*size
    gt=np.array([x['bbox'] for x in truth]);gleft=gt[:,:2];gright=gleft+gt[:,2:]
    area=np.maximum(right-left,0).prod(-1);garea=np.maximum(gright-gleft,0).prod(-1)
    intersection=np.maximum(np.minimum(right[:,None],gright[None])-np.maximum(left[:,None],gleft[None]),0).prod(-1)
    overlap=intersection/np.maximum(area[:,None]+garea[None]-intersection,1e-12)
    rows,cols=shared.linear_sum_assignment(-overlap)
    return {int(ids[row]):truth[col] for row,col in zip(rows,cols) if overlap[row,col]>=.5}

def classification_rates(counts,matched,total_gt):
    """Report both conditional classification and correct detections per GT.

    Unmatched GT contributes no correct classification to the all-GT rates.
    These fixed-geometry diagnostics do not replace COCO AP.
    """
    return dict(matched_fine_accuracy_pct=100*counts['fine_correct']/max(1,matched),
                matched_parent_accuracy_pct=100*counts['parent_correct']/max(1,matched),
                all_gt_fine_accuracy_pct=100*counts['fine_correct']/max(1,total_gt),
                all_gt_parent_accuracy_pct=100*counts['parent_correct']/max(1,total_gt))

def single_target_top1(predictions,truth_by_image,mapping,state_mapping=None):
    """Highest-score detection over ALL leaf classes, evaluated after fusion.

    Multi-target photos are excluded from this diagnostic and counted openly;
    matched-query and COCO metrics still cover their complete target sets.
    """
    top={}
    for row in predictions:
        iid=int(row['image_id'])
        if iid not in top or row['score']>top[iid]['score']:top[iid]=row
    count=fine=parent=state=joint=0;overlaps=[]
    for iid,truth in truth_by_image.items():
        if len(truth)!=1:continue
        target=truth[0];guess=top[iid];label=int(target['category_id']);pred=int(guess['category_id'])
        a,b=guess['bbox'],target['bbox']
        lo=np.maximum(a[:2],b[:2]);hi=np.minimum(np.array(a[:2])+a[2:],np.array(b[:2])+b[2:])
        inter=float(np.maximum(hi-lo,0).prod())
        union=max(0.,a[2])*max(0.,a[3])+max(0.,b[2])*max(0.,b[3])-inter
        overlap=inter/union if union>0 else 0.
        count+=1;fine+=int(pred==label);parent+=int(mapping[pred]==mapping[label])
        if state_mapping is not None:state+=int(state_mapping[pred]==state_mapping[label])
        joint+=int(pred==label and overlap>=.5);overlaps.append(overlap)
    return dict(single_target_images=count,multi_or_zero_target_images=len(truth_by_image)-count,
                top1_fine_accuracy_pct=100*fine/count if count else None,
                top1_parent_accuracy_pct=100*parent/count if count else None,
                top1_state_accuracy_pct=100*state/count if count and state_mapping is not None else None,
                top1_fine_and_iou50_accuracy_pct=100*joint/count if count else None,
                top1_mean_iou=float(np.mean(overlaps)) if count else None,
                top1_scope='global highest-score detection over all model leaves; single-GT photos only')


RUN_ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=Path('E:/ISW_DETR_external_data/CCCM_full_20261024')
COEFFICIENTS=dict(alpha=.5,beta=2.,eta=.2,observation_sigma_g=.25,class_std_floor_g=1.)


def state_taxonomy(categories):
    rows=sorted(categories,key=lambda row:int(row['id']))
    states=torch.tensor([int(row['state_id']) for row in rows],dtype=torch.long)
    if states.min()<0 or sorted(states.unique().tolist())!=list(range(7)):
        raise ValueError('Expected the seven source-native content states, 0..6')
    for parent in {int(row['identity_id']) for row in rows}:
        if sorted(int(row['state_id']) for row in rows if int(row['identity_id'])==parent)!=list(range(7)):
            raise ValueError('Each native container must have all seven source states')
    return states


def source_files(config=None):
    paths=[Path(__file__),COMMON,shared.REFINEMENT_PATH,
           shared.RUNTIME/'src/zoo/rtdetr/cvsr.py',
           shared.RUNTIME/'src/zoo/rtdetr/cfqd.py',
           shared.RUNTIME/'src/zoo/rtdetr/rtdetrv2_criterion.py',
           shared.RUNTIME/'src/zoo/rtdetr/rtdetr_postprocessor.py']
    if config is not None:paths.append(Path(config))
    return paths


def check_ready_and_lock(root,protocol_path,protocol,args):
    """Run only after complete file QA and the root experiment source lock."""
    ready_path=RUN_ROOT/'data_adapter/DATA_READY_QA.json'
    lock_path=RUN_ROOT/'EXPERIMENT_LOCK.json'
    if not ready_path.is_file() or not lock_path.is_file():
        raise ValueError('Full-data DATA_READY_QA.json and EXPERIMENT_LOCK.json are required before evaluation')
    ready=read(ready_path);lock=read(lock_path)
    expected=sum(int(value) for value in protocol['counts'].values())
    if int(ready.get('expected_images',-1))!=expected or int(ready.get('verified_images',-2))!=expected:
        raise ValueError('Full image inventory has not passed verification')
    for field in ('all_member_size_crc_sha256_passed','all_native_png_rgb_decode_passed',
                  'all_native_boxes_metadata_valid'):
        if ready.get(field) is not True:raise ValueError(f'Data readiness failed: {field}')
    if ready.get('source_protocol_sha256')!=sha(protocol_path):raise ValueError('Data-ready protocol hash differs')
    if ready.get('split_counts')!=protocol['counts']:raise ValueError('Data-ready split counts differ')
    for path,expected_hash in ready.get('output_sha256',{}).items():
        if sha(path)!=expected_hash:raise ValueError(f'Data-ready output changed: {path}')
    if ready.get('manifest_sha256')!=sha(ready_path.with_name('IMAGE_MANIFEST.json')):
        raise ValueError('Verified image manifest changed')
    if lock.get('source_protocol_sha256')!=sha(protocol_path):raise ValueError('Experiment source protocol differs')
    if Path(lock.get('data_root','')).resolve()!=root.resolve():raise ValueError('Experiment data root differs')
    if lock.get('scoring_methods')!=list(METHODS):raise ValueError('Experiment scoring methods differ')
    if lock.get('scoring_coefficients')!=COEFFICIENTS:raise ValueError('Experiment coefficients differ')
    if args.seed not in lock.get('seeds',[]):raise ValueError('Training seed not experiment-locked')
    hashes=lock.get('source_hashes',{})
    for path in source_files():
        if hashes.get(str(path))!=sha(path):raise ValueError(f'Experiment runtime hash absent/changed: {path}')
    config_hashes=lock.get('configs',{})
    if config_hashes.get(str(Path(args.config)))!=sha(args.config):
        raise ValueError('Selected checkpoint config is not experiment-locked')
    return ready_path,lock_path


def load_inputs(root,split,protocol):
    ann_path=root/f'annotations/{split}.json';ann=read(ann_path)
    images=sorted(ann['images'],key=lambda row:int(row['id']))
    mapping=taxonomy(ann['categories']);states=state_taxonomy(ann['categories'])
    if len(images)!=int(protocol['counts'][split]):raise ValueError('Split count differs from source protocol')
    if sorted(read(root/'annotations/train.json')['categories'],key=lambda row:int(row['id']))!=sorted(
            ann['categories'],key=lambda row:int(row['id'])):
        raise ValueError('Train/evaluation taxonomy mismatch')
    observations_path=root/f'observations/{split}.json'
    rows=read(observations_path);observations={int(row['image_id']):row for row in rows}
    if len(rows)!=len(observations) or set(observations)!={int(x['id']) for x in images}:
        raise ValueError('Mass observation inventory mismatch')
    for row in observations.values():
        if set(row)!={'image_id','mass_unit','records','telemetry_available'}:
            raise ValueError('Unexpected observation fields, including potential labels')
        if row['mass_unit']!='g' or row['telemetry_available'] is not False or not row['records']:
            raise ValueError('Expected positive static gram records without physical telemetry')
        if any(set(record)!={'record_id','mass_g'} for record in row['records']):
            raise ValueError('Forbidden owner or label field in fusion inputs')
        if len({str(record['record_id']) for record in row['records']})!=len(row['records']):
            raise ValueError('Duplicate static record IDs')
        if any(not np.isfinite(float(record['mass_g'])) or float(record['mass_g'])<=0 for record in row['records']):
            raise ValueError('Invalid native configuration mass')
    return ann_path,ann,images,mapping,states,observations_path,observations


def export_memmaps(cfg,checkpoint,images,root,device,batch_size,out,split):
    """Keep full 300-query outputs while avoiding whole-split tensor lists."""
    logits_path=out/f'{split}_raw_logits.npy';boxes_path=out/f'{split}_raw_boxes.npy'
    metadata_path=out/f'{split}_raw_metadata.json'
    expected=dict(checkpoint_sha256=sha(checkpoint),config_sha256=sha(cfg._external_config_path),
                  image_ids=[int(row['id']) for row in images],
                  image_sizes=[[row['width'],row['height']] for row in images],
                  shape_logits=[len(images),300,int(cfg.yaml_cfg['num_classes'])],
                  shape_boxes=[len(images),300,4])
    if metadata_path.exists():
        old=read(metadata_path)
        for key,value in expected.items():
            if old.get(key)!=value:raise ValueError(f'Raw cache provenance differs: {key}')
        if sha(logits_path)!=old['logits_sha256'] or sha(boxes_path)!=old['boxes_sha256']:
            raise ValueError('Raw cache changed')
        return logits_path,boxes_path,metadata_path,dict(old,cache_reused=True)
    model=cfg.model
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    use_ema='ema' in payload and 'module' in payload['ema']
    model.load_state_dict(payload['ema']['module'] if use_ema else payload['model'],strict=True)
    model=model.to(device).eval();size=cfg.yaml_cfg['eval_spatial_size']
    logits=np.lib.format.open_memmap(logits_path,mode='w+',dtype=np.float32,shape=tuple(expected['shape_logits']))
    boxes=np.lib.format.open_memmap(boxes_path,mode='w+',dtype=np.float32,shape=tuple(expected['shape_boxes']))
    started=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(images),batch_size):
            batch=images[offset:offset+batch_size];inputs=[]
            for photo in batch:
                with Image.open(root/photo['file_name']) as source:
                    if source.size!=(photo['width'],photo['height']):raise ValueError('Source image dimensions changed')
                    image=source.convert('RGB').resize((int(size[1]),int(size[0])),Image.Resampling.BILINEAR)
                    inputs.append(shared.TF.to_tensor(image))
            value=model(torch.stack(inputs).to(device))
            raw=value['pred_logits'].detach().float().cpu().numpy()
            coords=value['pred_boxes'].detach().float().cpu().numpy()
            if not np.isfinite(raw).all() or not np.isfinite(coords).all():raise ValueError('Nonfinite raw outputs')
            logits[offset:offset+len(batch)]=raw;boxes[offset:offset+len(batch)]=coords
            if offset%100==0:print('RGB export',offset,'/',len(images),flush=True)
    logits.flush();boxes.flush();del logits,boxes
    metadata=dict(expected,checkpoint_state='ema.module' if use_ema else 'model',
                  precision='FP32 inference, no autocast',seconds=time.perf_counter()-started,
                  logits_sha256=sha(logits_path),boxes_sha256=sha(boxes_path))
    write(metadata_path,metadata)
    model.to('cpu');del payload,value,inputs
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    gc.collect()
    return logits_path,boxes_path,metadata_path,metadata


def fuse_memmaps(logits,boxes,images,observations,priors,mapping,out,split):
    marker=out/f'{split}_fusion_manifest.json'
    paths={key:out/f'{split}_{key}_refined.npy' for key in METHODS}
    assignment_paths={key:out/f'{split}_{key}_assignments.npy' for key in METHODS if key not in ('rgb','gaussian_broadcast')}
    if marker.exists():
        previous=read(marker)
        for path,expected_hash in previous['files_sha256'].items():
            if sha(path)!=expected_hash:raise ValueError('Completed fusion cache changed')
        return paths,assignment_paths,Path(previous['candidate_file'])
    values={key:np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=logits.shape) for key,path in paths.items()}
    assignments={key:np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=logits.shape[:2])
                 for key,path in assignment_paths.items()}
    diagnostics=[]
    with torch.inference_mode():
        for index,photo in enumerate(images):
            records=observations[int(photo['id'])]['records']
            output,diagnostic,edges=shared.score_one(torch.from_numpy(logits[index].copy()),
                torch.from_numpy(boxes[index].copy()),[float(row['mass_g']) for row in records],priors,mapping)
            diagnostic['image_id']=int(photo['id']);diagnostics.append(diagnostic)
            for key in METHODS:values[key][index]=output[key].numpy()
            for key in assignments:assignments[key][index]=edges[key].numpy()
            if index%250==0:print('Static scoring',index,'/',len(images),flush=True)
    for value in [*values.values(),*assignments.values()]:value.flush()
    values.clear();assignments.clear();gc.collect()
    candidate_file=out/f'{split}_candidates.json';write(candidate_file,diagnostics)
    write(marker,dict(candidate_file=str(candidate_file),full_query_shape=list(logits.shape),
        telemetry_gate_tested=False,static_numeric_eligibility_only=True,
        files_sha256={str(path):sha(path) for path in [*paths.values(),*assignment_paths.values(),candidate_file]}))
    return paths,assignment_paths,candidate_file


def classification_diagnostics(refined,fixed,assignment,logits,boxes,images,truth_by_image,observations,owners,mapping,states):
    counts=dict(fine_correct=0,parent_correct=0,state_correct=0,corrected_vs_fixed=0,harmed_vs_fixed=0)
    association=dict(assigned=0,geometry_valid=0,strict_owner_correct=0,mass_equivalent_correct=0)
    matched=0
    for index,photo in enumerate(images):
        truth=truth_by_image[int(photo['id'])];events=len(observations[int(photo['id'])]['records'])
        pairs=geometry(torch.from_numpy(logits[index].copy()),torch.from_numpy(boxes[index].copy()),photo,truth,events)
        matched+=len(pairs)
        for query,target in pairs.items():
            label=int(target['category_id']);predicted=int(np.argmax(refined[index,query]))
            fixed_label=int(np.argmax(fixed[index,query]))
            counts['fine_correct']+=int(predicted==label)
            counts['parent_correct']+=int(mapping[predicted]==mapping[label])
            counts['state_correct']+=int(states[predicted]==states[label])
            counts['corrected_vs_fixed']+=int(predicted==label and fixed_label!=label)
            counts['harmed_vs_fixed']+=int(predicted!=label and fixed_label==label)
        if assignment is None:continue
        owner_list=owner_entries(owners[int(photo['id'])])
        actual={int(owner['annotation_id']):float(owner['mass_g']) for owner in owner_list}
        for query,event in enumerate(assignment[index]):
            if int(event)<0:continue
            association['assigned']+=1
            if query not in pairs:continue
            association['geometry_valid']+=1;target=pairs[query];owner=owner_list[int(event)]
            association['strict_owner_correct']+=int(int(target['id'])==int(owner['annotation_id']))
            association['mass_equivalent_correct']+=int(abs(actual[int(target['id'])]-float(owner['mass_g']))<=.001)
    return counts,association,matched


def compact_json(path,value):
    with Path(path).open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def evaluate_methods(cfg,out,split,ann_path,ann,images,mapping,states,observations,root,logits,boxes,paths,assignment_paths):
    # No GT owner labels are loaded until fusion outputs for every method exist.
    owner_path=root/f'evaluation_sidecars/{split}_owners.json'
    owner_rows=read(owner_path);owners={int(row['image_id']):row for row in owner_rows}
    if len(owners)!=len(owner_rows) or set(owners)!=set(observations):raise ValueError('Owner inventory differs')
    for iid,item in owners.items():
        entries=owner_entries(item);records=observations[iid]['records']
        if len(entries)!=len(records) or any(str(a['record_id'])!=str(b['record_id']) for a,b in zip(entries,records)):
            raise ValueError('Owner record axis differs')
    truth_by_image={int(row['id']):[] for row in images}
    for row in ann['annotations']:truth_by_image[int(row['image_id'])].append(row)
    parents=len(mapping.unique());coarse=dict(ann)
    coarse['categories']=[dict(id=p,name=f'native_container_{p}') for p in range(parents)]
    coarse['annotations']=[dict(row,category_id=int(mapping[int(row['category_id'])])) for row in ann['annotations']]
    coarse_path=out/f'{split}_coarse_gt.json';write(coarse_path,coarse)
    post=cfg.postprocessor.cpu().eval();parent_post=copy.deepcopy(post);parent_post.num_classes=parents
    fixed=np.load(paths['original_original'],mmap_mode='r')
    metrics={};association={};artifact_hashes={};image_ids=[int(row['id']) for row in images]
    for key in METHODS:
        refined=np.load(paths[key],mmap_mode='r')
        edges=np.load(assignment_paths[key],mmap_mode='r') if key in assignment_paths else None
        counts,ownership,matched=classification_diagnostics(refined,fixed,edges,logits,boxes,images,
            truth_by_image,observations,owners,mapping,states)
        predictions=[]
        with torch.inference_mode():
            for index,photo in enumerate(images):
                predictions.extend(shared.as_coco(post,torch.from_numpy(refined[index].copy()),
                    torch.from_numpy(boxes[index].copy()),photo['id'],[photo['width'],photo['height']]))
        prediction_path=out/f'{split}_{key}_predictions.json';compact_json(prediction_path,predictions)
        row=shared.coco_metrics(ann_path,predictions,image_ids,out/f'{split}_{key}_coco.txt')
        row.update(**classification_rates(counts,matched,len(ann['annotations'])),**counts,
            matched_state_accuracy_pct=100*counts['state_correct']/max(1,matched),
            all_gt_state_accuracy_pct=100*counts['state_correct']/max(1,len(ann['annotations'])),
            **single_target_top1(predictions,truth_by_image,mapping,states))
        del predictions;gc.collect()
        # One method and one label space in memory at a time.
        predictions=[]
        with torch.inference_mode():
            for index,photo in enumerate(images):
                scores=torch.from_numpy(refined[index].copy())
                parent=torch.stack([scores[:,mapping==p].max(1).values for p in range(parents)],1)
                predictions.extend(shared.as_coco(parent_post,parent,torch.from_numpy(boxes[index].copy()),
                    photo['id'],[photo['width'],photo['height']]))
        parent_path=out/f'{split}_{key}_coarse_predictions.json';compact_json(parent_path,predictions)
        parent_metrics=shared.coco_metrics(coarse_path,predictions,image_ids,out/f'{split}_{key}_coarse_coco.txt')
        row.update(parent_AP=parent_metrics['AP'],parent_AP50=parent_metrics['AP50'])
        metrics[key]=row
        if key in assignment_paths:association[key]=ownership
        artifact_hashes.update({str(path):sha(path) for path in (prediction_path,parent_path)})
        print(key,'AP',row['AP'],'parent AP',row['parent_AP'],'state top1',row['top1_state_accuracy_pct'],flush=True)
        del predictions,refined,edges;gc.collect()
    del fixed
    for row in metrics.values():
        row['delta_AP_vs_fixed_pp']=row['AP']-metrics['original_original']['AP']
        row['delta_AP_vs_rgb_pp']=row['AP']-metrics['rgb']['AP']
    return metrics,association,matched,artifact_hashes,owner_path


def qa_only(args):
    """Metadata and tiny CPU scoring checks; never evaluate public test outputs."""
    root=Path(args.data_root);protocol_path=RUN_ROOT/'data_adapter/SOURCE_PROTOCOL.json';protocol=read(protocol_path)
    source_paths,chain=source_provenance(protocol,protocol_path)
    _,ann,images,mapping,states,_,observations=load_inputs(root,'train',protocol)
    priors,prior_path=train_priors(root,len(mapping))
    torch.manual_seed(2026102404)
    logits=torch.randn(300,len(mapping));boxes=torch.rand(300,4)
    sample=observations[int(images[0]['id'])]
    result,diagnostic,assignments=shared.score_one(logits,boxes,[x['mass_g'] for x in sample['records']],priors,mapping)
    assert tuple(result)==METHODS and all(torch.isfinite(row).all() for row in result.values())
    assert torch.equal(result['rgb'],logits)
    outside=torch.ones(300,dtype=torch.bool);outside[diagnostic['candidate_indices']]=False
    assert all(torch.equal(row[outside],logits[outside]) for row in result.values())
    config=None
    if args.config:
        cfg=shared.YAMLConfig(args.config,PResNet={'pretrained':False})
        assert int(cfg.yaml_cfg['num_classes'])==len(mapping)
        config=dict(path=str(Path(args.config)),sha256=sha(args.config),
                    model=cfg.yaml_cfg['model'],input_size=cfg.yaml_cfg['eval_spatial_size'])
    report=dict(status='PASS',scope='train metadata/source chain and random CPU tensors only; no public test inference or metrics',
        source_provenance=chain,counts=protocol['counts'],train_images=len(images),
        trained_classes=len(mapping),parents=len(mapping.unique()),states=len(states.unique()),
        methods=list(METHODS),coefficients=COEFFICIENTS,all_methods_finite=True,noncandidate_outputs_exact=True,
        train_only_priors=priors,prior_source_sha256=sha(prior_path),config=config,
        full_data_ready_file_exists=(RUN_ROOT/'data_adapter/DATA_READY_QA.json').exists(),
        experiment_lock_exists=(RUN_ROOT/'EXPERIMENT_LOCK.json').exists(),
        telemetry_gate_tested=False,actual_test_performance_observed_by_this_qa=False,
        source_hashes={str(path):sha(path) for path in source_files(args.config)})
    destination=Path(args.output) if args.output else RUN_ROOT/'runtime/STATIC_QA.json'
    write(destination,report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('train_only_priors','source_hashes')},indent=2))


def run(args):
    if args.qa_only:return qa_only(args)
    for field in ('checkpoint','config','output','seed','visual_variant'):
        if getattr(args,field) is None:raise ValueError(f'--{field.replace("_","-")} is required for evaluation')
    torch.set_num_threads(2);root=Path(args.data_root);out=Path(args.output)
    protocol_path=RUN_ROOT/'data_adapter/SOURCE_PROTOCOL.json';protocol=read(protocol_path)
    source_paths,chain=source_provenance(protocol,protocol_path)
    ready_path,experiment_path=check_ready_and_lock(root,protocol_path,protocol,args)
    ann_path,ann,images,mapping,states,observation_path,observations=load_inputs(root,args.split,protocol)
    priors,prior_path=train_priors(root,len(mapping))
    cfg=shared.YAMLConfig(args.config,PResNet={'pretrained':False});cfg._external_config_path=str(args.config)
    if int(cfg.yaml_cfg['num_classes'])!=len(mapping):raise ValueError('Class count differs')
    actual_is=cfg.yaml_cfg['model']=='CFQDCVSRRTDETR'
    if actual_is!=(args.visual_variant=='is'):raise ValueError('Visual variant does not describe model config')
    if actual_is and (cfg.yaml_cfg['CFQDRTDETRTransformerv2'].get('cfqd_num_states')!=7 or
                      cfg.yaml_cfg['CFQDCVSRRTDETR'].get('enable_cvsr',False)):
        raise ValueError('Expected seven-state visual I-S without physical-transaction simulation')
    out.mkdir(parents=True,exist_ok=True)
    if (out/f'{args.split}_summary.json').exists():raise ValueError('Completed evaluation exists; preserve all results')
    owner_path=root/f'evaluation_sidecars/{args.split}_owners.json'
    locked_paths=list(dict.fromkeys(source_files(args.config)+source_paths+
        [Path(args.checkpoint),ann_path,observation_path,owner_path,prior_path,root/'annotations/train.json',
         root/'taxonomy.json',ready_path,experiment_path]))
    evaluation_lock=dict(seed=args.seed,visual_variant=args.visual_variant,split=args.split,
        methods=list(METHODS),coefficients=COEFFICIENTS,
        input_source_sha256={str(path):sha(path) for path in locked_paths})
    lock_path=out/f'{args.split}_protocol_lock.json'
    if lock_path.exists() and read(lock_path)!=evaluation_lock:raise ValueError('Incomplete evaluation lock changed')
    write(lock_path,evaluation_lock);write(out/'train_only_mass_priors.json',priors)
    logits_path,boxes_path,raw_meta,export_meta=export_memmaps(cfg,args.checkpoint,images,root,
        args.device,args.batch_size,out,args.split)
    logits=np.load(logits_path,mmap_mode='r');boxes=np.load(boxes_path,mmap_mode='r')
    paths,assignment_paths,candidate_path=fuse_memmaps(logits,boxes,images,observations,priors,mapping,out,args.split)
    metrics,association,matched,prediction_hashes,owner_path=evaluate_methods(cfg,out,args.split,
        ann_path,ann,images,mapping,states,observations,root,logits,boxes,paths,assignment_paths)
    summary=dict(status='complete: full visual branch plus controlled static-mass fusion',
        created_utc=datetime.now(timezone.utc).isoformat(),split=args.split,training_seed=args.seed,
        visual_variant=args.visual_variant,I_S_auxiliary_head_trained=actual_is,
        I_S_state_count=7 if actual_is else None,images=len(images),total_gt=len(ann['annotations']),
        trained_leaf_classes=len(mapping),parent_classes=len(mapping.unique()),state_classes=len(states.unique()),
        source_recording_groups=None,source_recording_ids_available=False,
        native_views=sorted({int(row['native_view']) for row in images}),
        physical_objects_shared_across_views=True,known_configurations_shared_across_views=True,
        previous_pilot_test_observed=True,blind_test_claim=False,
        native_mass_type=protocol['mass_definition'],telemetry_gate_tested=False,
        full_real_weighing_transaction_validation=False,static_numeric_eligibility_only=True,
        multi_record_association_tested=False,confidence_intervals=False,
        source_protocol_scope=dict(split_views=protocol['split_views'],task='known-container cross-view'),
        checkpoint_selection='RGB-only dev AP best under experiment-locked branch training; no mass-score test selection',
        classification_geometry_matched=matched,geometry_recall_pct=100*matched/max(1,len(ann['annotations'])),
        classification_denominators=dict(matched='shared RGB-candidate geometric pairs at IoU>=0.5',
            all_gt='all targets, unmatched contribute no correct predictions',
            state='seven native content states evaluated even when parent identity is wrong'),
        metrics=metrics,association=association,source_provenance=chain,
        sources=evaluation_lock['input_source_sha256'],prediction_sha256=prediction_hashes,
        raw_cache=dict(logits=str(logits_path),boxes=str(boxes_path),metadata=str(raw_meta)),
        refined_cache={key:str(path) for key,path in paths.items()},export=export_meta)
    # Verify all locked code and inputs again before publishing the summary.
    if any(sha(path)!=expected for path,expected in evaluation_lock['input_source_sha256'].items()):
        raise ValueError('Locked source/input changed while evaluating')
    write(out/f'{args.split}_summary.json',summary)
    lines=['# C-CCM full visual branch with static-mass fusion','',
        f'Visual branch: {args.visual_variant}; seed {args.seed}; split {args.split}.',
        '','| Method | Fine AP | Container AP | Matched state % | Top1 fine % | Top1 state % | Delta AP vs fixed |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,row in metrics.items():
        lines.append(f"| {key} | {row['AP']:.4f} | {row['parent_AP']:.4f} | {row['matched_state_accuracy_pct']:.2f} | {row['top1_fine_accuracy_pct']:.2f} | {row['top1_state_accuracy_pct']:.2f} | {row['delta_AP_vs_fixed_pp']:+.4f} |")
    (out/f'{args.split}_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
