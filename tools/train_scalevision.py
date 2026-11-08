"""Portable B0 -> three warm-start visual detectors -> I-S auxiliary training.

The historical frozen carriers contain unrelated heads. Their base tensors
equal the warm checkpoint EMA; those unused heads need not be retrained.
This entry preserves the original stage budgets and validation selection.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'runtime'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def relocated(value,dataset):
    if isinstance(value,dict):return {k:relocated(v,dataset) for k,v in value.items()}
    if isinstance(value,list):return [relocated(v,dataset) for v in value]
    if isinstance(value,str):
        text=value.replace('\\','/')
        for old,new in [('D:/MY_DATA/09-Dataset',dataset),
                        ('D:/MY_DATA/02-project/sci_project/RT_DETR/Official/RT-DETR/rtdetrv2_pytorch',NATIVE)]:
            if text.casefold().startswith(old.casefold()+'/'):return (new/text[len(old)+1:]).as_posix()
    return value


def stage(args,kind,seed,output,initializer):
    config_path=ROOT/f'configs/scalevision/training_{kind}.yml'
    config=relocated(yaml.safe_load(config_path.read_text(encoding='utf8')),Path(args.dataset_root).resolve())
    folder=output/f'{kind}_seed{seed}'
    if folder.exists() and any(folder.iterdir()):raise ValueError('Use a fresh stage directory: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    config['output_dir']=folder.as_posix()
    hashes={str(config_path):sha(config_path)}
    input_spec=ROOT/'configs/scalevision/training_inputs.json'
    fixed=read(input_spec)
    hashes[str(input_spec)]=sha(input_spec)
    for relative,expected in fixed['fixed_delivered_input_sha256'].items():
        path=Path(args.dataset_root).resolve()/'ScaleVision-MM-V3'/relative
        if not path.is_file() or sha(path)!=expected:raise ValueError('Fixed delivered training input changed: '+str(path))
    if kind=='b0':
        if not args.backbone_pretrained:raise ValueError('B0 requires --backbone-pretrained (original Paddle-converted ResNet18-vd), not the full COCO detector')
        backbone=Path(args.backbone_pretrained).resolve()
        if not backbone.is_file():raise FileNotFoundError(backbone)
        if sha(backbone)!=fixed['backbone_sha256']:raise ValueError('Backbone weights differ from the recovered reference initializer')
        config['PResNet']['pretrained']=backbone.as_posix();hashes[str(backbone)]=sha(backbone)
        if initializer is not None:raise ValueError('Historical B0 has no full-detector tuning checkpoint')
    else:
        if initializer is None or not initializer.is_file():raise FileNotFoundError('Supply a valid initializer for '+kind)
        config['PResNet']['pretrained']=False
        hashes[str(initializer)]=sha(initializer)
    for key in ['train_dataloader','val_dataloader']:
        dataset=config[key]['dataset']
        annotation=Path(dataset['ann_file'])
        if not annotation.is_file() or not Path(dataset['img_folder']).is_dir():raise FileNotFoundError('Missing fixed ScaleVision data: '+str(annotation))
        hashes[str(annotation)]=sha(annotation)
        if args.smoke:
            source=read(annotation);images=source['images'][:8];ids={int(r['id']) for r in images}
            subset={**source,'images':images,'annotations':[r for r in source['annotations'] if int(r['image_id']) in ids]}
            path=folder/(key+'_smoke.json');write(path,subset);dataset['ann_file']=path.as_posix()
            config[key]['num_workers']=0
            config[key]['total_batch_size']=2
    if kind=='is':
        taxonomy=Path(config['CFQDRTDETRTransformerv2']['taxonomy_path'])
        hashes[str(taxonomy)]=sha(taxonomy)
    if args.smoke:
        config['epoches']=1;config['checkpoint_freq']=1
        config.setdefault('early_stop',{})['enabled']=False
    local=folder/'resolved_config.yml';local.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf8')
    hashes[str(local)]=sha(local)
    lock=dict(stage=kind,seed=seed,stage_epoch_budget=config['epoches'],smoke_only=args.smoke,
        actual_initialization='backbone pretrained only' if kind=='b0' else str(initializer),
        tuning_prefers_EMA_when_present=True,input_sha256=hashes,
        historical_chain_scope='Recovered actual B0-to-warm execution commands and tensor-verified warm-EMA-to-I-S frozen base. No claim to reconstruct unrecorded earlier search attempts.')
    write(folder/'TRAINING_INPUT_LOCK.json',lock)
    if args.prepare_only:
        return folder/'best.pth'
    cmd=[sys.executable,'-X','utf8','-B','tools/train.py','-c',str(local),'-d',args.device,'--seed',str(seed)]
    if kind!='b0':cmd+=['-t',str(initializer)]
    if args.device.startswith('cuda'):cmd+=['--use-amp']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with (folder/'training_console.log').open('w',encoding='utf8') as stream:
        result=subprocess.run(cmd,cwd=NATIVE,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Training failed; see '+str(folder/'training_console.log'))
    best=folder/'best.pth'
    if not best.is_file():raise RuntimeError('Training produced no validation-selected best checkpoint')
    if any(sha(path)!=h for path,h in hashes.items()):raise RuntimeError('An input changed during training')
    import torch
    payload=torch.load(best,map_location='cpu',weights_only=True)
    log=[read_line for read_line in (folder/'log.txt').read_text(encoding='utf8').splitlines() if read_line.strip()]
    epochs=[json.loads(line) for line in log]
    report=dict(status='complete',stage=kind,seed=seed,smoke_only=args.smoke,
        checkpoint=str(best),checkpoint_sha256=sha(best),best_epoch=payload['last_epoch'],
        completed_epochs=len(epochs),original_epoch_budget=lock['stage_epoch_budget'],input_lock_sha256=sha(folder/'TRAINING_INPUT_LOCK.json'))
    write(folder/'TRAINING_RESULT.json',report)
    print(json.dumps(report),flush=True)
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['b0','warm','is','all'],required=True)
    p.add_argument('--dataset-root',required=True,help='Contains ScaleVision-MM-V3')
    p.add_argument('--output',required=True)
    p.add_argument('--initializer',help='Explicit B0 checkpoint for warm, or warm/carrier checkpoint for is')
    p.add_argument('--backbone-pretrained')
    p.add_argument('--seed',type=int,choices=[42,3407,2026],default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='8 images and 1 epoch; diagnostic only, never a paper training run')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage=='all' and args.prepare_only:p.error('Prepare stages separately; later initializers do not exist before actual preceding training')
    if args.stage=='b0' and args.seed!=42:p.error('The recovered shared B0 was trained with seed42')
    init=Path(args.initializer).resolve() if args.initializer else None
    if args.stage=='all':
        b0=stage(args,'b0',42,out,None)
        for seed in [42,3407,2026]:
            warm=stage(args,'warm',seed,out,b0)
            stage(args,'is',seed,out,warm)
    else:stage(args,args.stage,args.seed,out,init)


if __name__=='__main__':main()
