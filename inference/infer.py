"""Raw RGB + observed transaction inference. No annotations/GT argument exists."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(ROOT/'adapters'))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as TF
from src.core import YAMLConfig
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from native_detector import StateNeutralDetector

TRANSACTION_KEYS = ('weight_tokens_raw','weight_mask','event_order','scene_total_weight_g',
                    'empty_baseline_g','scene_total_available','scene_total_independent')

def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def image_tensor(image):
    resized=TF.resize(image,[640,640],interpolation=InterpolationMode.BILINEAR,antialias=True)
    return TF.pil_to_tensor(resized).float()/255.

def load_transaction(path,device):
    record=read(path)
    if not isinstance(record,dict): raise ValueError('Transaction must be a JSON object.')
    allowed=set(TRANSACTION_KEYS)|{'schema_version','measurement_provenance'}
    extra=set(record)-allowed
    if extra: raise ValueError(f'Unknown transaction keys (labels/boxes are forbidden): {sorted(extra)}')
    if 'weight_tokens_raw' not in record: raise ValueError('Transaction requires weight_tokens_raw, even when empty.')
    raw=record['weight_tokens_raw']
    if not isinstance(raw,list) or any(not isinstance(row,list) or len(row)!=7 for row in raw):
        raise ValueError('weight_tokens_raw must contain E rows of seven ordered measurements.')
    if len(raw)>8: raise ValueError('This protocol supports at most eight events; no silent truncation.')
    events=len(raw)
    record.setdefault('weight_mask',[True]*events)
    if not isinstance(record['weight_mask'],list) or len(record['weight_mask'])!=events:
        raise ValueError('weight_mask must have length E.')
    if any(type(value) is not bool for value in record['weight_mask']): raise ValueError('weight_mask entries must be JSON booleans.')
    if 'event_order' in record and len(record['event_order'])!=events: raise ValueError('event_order must have length E.')
    for name in ('scene_total_available','scene_total_independent'):
        if name in record and type(record[name]) is not bool: raise ValueError(f'{name} must be a JSON boolean.')
    result={}
    for key in TRANSACTION_KEYS:
        if key not in record or record[key] is None: continue
        dtype=torch.bool if key in ('weight_mask','scene_total_available','scene_total_independent') else torch.float32
        value=torch.tensor(record[key],dtype=dtype,device=device)
        if key=='weight_tokens_raw': value=value.reshape(events,7)
        result[key]=value.unsqueeze(0)
    return result,record

def build_model(seed,device,checkpoint=None):
    config_path=ROOT/f'configs/inference_seed{seed}.yml'
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        cfg=YAMLConfig(str(config_path))
        cfg.yaml_cfg['use_amp']=False
        cfg.yaml_cfg['use_ema']=False
        cfg.yaml_cfg['PResNet']['pretrained']=False
        cfg.yaml_cfg['CFQDRTDETRTransformerv2']['taxonomy_path']=str(ROOT/'inputs/taxonomy.json')
        cfg.yaml_cfg['CFQDCVSRRTDETR']['physics_stats_path']=str(ROOT/'inputs/prior.json')
        model=cfg.model
        checkpoint_path=resolve(checkpoint) if checkpoint else ROOT/f'weights/detector_seed{seed}.pth'
        if not checkpoint_path.is_file(): raise FileNotFoundError('Use --checkpoint to supply external trained weights: '+str(checkpoint_path))
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        loaded=model.load_state_dict(checkpoint['model'],strict=False)
        allowed={'decoder.cfqd_paired_parent','cvsr.class_mean_g','cvsr.class_std_g'}
        if set(loaded.missing_keys)-allowed or loaded.unexpected_keys:
            raise RuntimeError(f'Unexpected checkpoint interface: {loaded}')
        model.eval().requires_grad_(False).to(device)
        mapping=read(ROOT/'inputs/taxonomy.json')['leaf_to_parent']
        # Never move this whole wrapper to CUDA: its fusion bridge stays on CPU.
        wrapped=StateNeutralDetector(model,mapping,.5,copy_detector=False).eval()
    return wrapped,{'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,
        'build_log':capture.getvalue(),'detector_sha256':sha(checkpoint_path),
        'config_sha256':sha(config_path),'prior_sha256':sha(ROOT/'inputs/prior.json'),
        'taxonomy_sha256':sha(ROOT/'inputs/taxonomy.json')}

def prediction_payload(logits,boxes,size):
    post=RTDETRPostProcessor(num_classes=28,num_top_queries=300,remap_mscoco_category=False).to(logits.device)
    result=post({'pred_logits':logits,'pred_boxes':boxes},size)[0]
    return {key:value.cpu().tolist() for key,value in result.items()}

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--transaction')
    parser.add_argument('--seed',type=int,choices=(42,3407,2026),default=42)
    parser.add_argument('--checkpoint',help='External trained checkpoint; weights are not included in this code-only package')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--output',default='outputs/predictions.json')
    parser.add_argument('--save-queries')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    image_path=resolve(args.image)
    image=Image.open(image_path).convert('RGB')
    modal,record=(None,None) if not args.transaction else load_transaction(resolve(args.transaction),'cpu')
    wrapped,provenance=build_model(args.seed,device,args.checkpoint)
    x=image_tensor(image)[None].to(device)
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    result=wrapped(x,modal_inputs=modal)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    # Native postprocessor multiplies xyxy by [width,height,width,height].
    size=torch.tensor([[image.width,image.height]],device=device)
    visual=result['pred_logits_cfqd'];final=result['pred_logits']
    gate=result['state_integrity_gate'];reasons=result['state_integrity_rejection_code']
    code=int(reasons[0]);physics=wrapped.bridge.upstream
    payload={'schema_version':2,'method':'state-contrast ISW-DETR','seed':args.seed,
        'visual_device':str(device),'fusion_device':'cpu','precision':'float32','readout_alpha':.5,
        'input_image_sha256':sha(image_path),'input_transaction_sha256':None if not args.transaction else sha(resolve(args.transaction)),
        'ground_truth_used':False,'all_parameters_frozen':all(not p.requires_grad for p in wrapped.parameters()),
        'checkpoint_provenance':provenance,'integrity_gate':bool(gate[0]),'rejection_code':code,
        'rejection_reasons':[key for key,value in physics.REJECTION_BITS.items() if code&value],
        'scene_total_independent':False if record is None else bool(record.get('scene_total_independent',False)),
        'exact_visual_fallback':bool(torch.equal(final,visual)) if not bool(gate[0]) else None,
        'candidate_count':int(result['state_candidate_mask'][0].sum()),'image_size':[image.width,image.height],
        'forward_elapsed_s':elapsed,'detections':prediction_payload(final,result['pred_boxes'],size),
        'visual_detections':prediction_payload(visual,result['pred_boxes'],size)}
    if args.save_queries:
        path=resolve(args.save_queries);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,visual_logits=visual.cpu().numpy(),isw_logits=final.cpu().numpy(),
            boxes=result['pred_boxes'].cpu().numpy(),candidate_mask=result['state_candidate_mask'].cpu().numpy(),
            assigned_event=result['state_assigned_event'].cpu().numpy())
    path=resolve(args.output);write(path,payload)
    print(json.dumps({'output':str(path),'integrity_gate':payload['integrity_gate'],
        'exact_visual_fallback':payload['exact_visual_fallback'],'detections':len(payload['detections']['scores'])}))

if __name__=='__main__':main()
