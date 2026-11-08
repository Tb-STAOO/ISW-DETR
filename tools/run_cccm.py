"""Portable C-CCM input check, original-budget training, or seven-score evaluation.

Datasets and checkpoints are external inputs. Core model/scoring formulas
are unchanged; runtime configurations and outputs are created separately.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
from package_paths import ROOT,PROJECT,RT,NATIVE,DATA,RUN,relocated,local_config,resolve

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def evaluator():
    p=ROOT/'evaluation/cccm_static.py'
    spec=importlib.util.spec_from_file_location('packaged_cccm_original_evaluator',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def check_inputs():
    lock=read(RUN/'EXPERIMENT_LOCK.json');checks=[]
    for rel,expected in lock['input_hashes'].items():
        p=DATA/rel
        checks.append(dict(local=str(p),exists=p.is_file(),matches=p.is_file() and sha(p)==expected))
    counts={};missing=[]
    for name,expected in lock['counts'].items():
        ann=read(DATA/f'annotations/{name}.json');counts[name]=len(ann['images'])
        if counts[name]!=expected:raise ValueError('Split size changed: '+name)
        missing.extend(str(DATA/r['file_name']) for r in ann['images'] if not (DATA/r['file_name']).is_file())
    result=dict(passed=all(c['matches'] for c in checks) and not missing,counts=counts,source_and_input_hash_checks=checks,missing_images=missing)
    if not result['passed']:raise ValueError('External C-CCM inputs differ from the fixed prepared data: '+json.dumps(result))
    m=evaluator();protocol=dict(counts=lock['counts'])
    for split in ['train','dev','test']:m.load_inputs(DATA,split,protocol)
    return m,protocol,result

def train_one(args,variant,seed,output_root):
    config=local_config(variant,seed,output_root)
    out=output_root/f'{variant}_seed{seed}'
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh training output directory: '+str(out))
    init=Path(args.initializer).resolve() if args.initializer else (
        Path(args.pretrained) if variant=='base' else
        output_root/f'base_seed{seed}/best.pth' if (output_root/f'base_seed{seed}/best.pth').exists() else DATA/f'runs/base_seed{seed}/best.pth')
    if not init.is_file():raise ValueError('Missing initializer: '+str(init))
    out.mkdir(parents=True,exist_ok=True)
    write(out/'PACKAGE_TRAINING_INPUT_LOCK.json',dict(config_sha256=sha(config),initializer=str(init),initializer_sha256=sha(init),variant=variant,seed=seed,source_lock=str(RUN/'EXPERIMENT_LOCK.json'),source_lock_sha256=sha(RUN/'EXPERIMENT_LOCK.json')))
    cmd=[sys.executable,'-X','utf8','-B','tools/train.py','-c',str(config),'--seed',str(seed),'-d',args.device,'-t',str(init)]
    if args.device.startswith('cuda'):cmd.append('--use-amp')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with (out/'package_training_console.log').open('w',encoding='utf8') as log:
        status=subprocess.run(cmd,cwd=NATIVE,env=env,stdout=log,stderr=subprocess.STDOUT)
    if status.returncode:raise RuntimeError('Training failed; inspect '+str(out/'package_training_console.log'))

def evaluate(args,m,protocol):
    import numpy as np
    import torch
    out=resolve(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Evaluation requires a fresh output directory')
    if shutil.disk_usage(out).free<4*1024**3:raise ValueError('Seven-score evaluation needs at least 4GiB free; select a larger output drive')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=local_config(args.variant,args.seed)
    checkpoint=resolve(args.checkpoint) if args.checkpoint else DATA/f'runs/{args.variant}_seed{args.seed}/best.pth'
    ann_path,ann,images,mapping,states,obs_path,observations=m.load_inputs(DATA,args.split,protocol)
    priors,prior_path=m.train_priors(DATA,len(mapping))
    cfg=m.shared.YAMLConfig(str(config),PResNet={'pretrained':False});cfg._external_config_path=str(config)
    if int(cfg.yaml_cfg['num_classes'])!=42:raise ValueError('Expected 42 leaf classes')
    actual_is=cfg.yaml_cfg['model']=='CFQDCVSRRTDETR'
    if actual_is!=(args.variant=='is'):raise ValueError('Visual branch/config mismatch')
    sources=list(dict.fromkeys(m.source_files(config)+[checkpoint,ann_path,obs_path,prior_path,DATA/'taxonomy.json']))
    lock=dict(seed=args.seed,visual_variant=args.variant,split=args.split,methods=list(m.METHODS),coefficients=m.COEFFICIENTS,source_sha256={str(p):sha(p) for p in sources},history_lock_sha256=sha(RUN/'EXPERIMENT_LOCK.json'),relocation_only_wrapper=True)
    write(out/'PACKAGED_EXECUTION_LOCK.json',lock)
    lp,bp,meta,export=m.export_memmaps(cfg,checkpoint,images,DATA,args.device,args.batch_size,out,args.split)
    logits=np.load(lp,mmap_mode='r');boxes=np.load(bp,mmap_mode='r')
    paths,assignments,candidates=m.fuse_memmaps(logits,boxes,images,observations,priors,mapping,out,args.split)
    metrics,association,matched,prediction_hashes,owner_path=m.evaluate_methods(cfg,out,args.split,ann_path,ann,images,mapping,states,observations,DATA,logits,boxes,paths,assignments)
    if any(sha(Path(p))!=h for p,h in lock['source_sha256'].items()):raise ValueError('Input changed during evaluation')
    result=dict(status='complete',seed=args.seed,visual_variant=args.variant,split=args.split,images=len(images),metrics=metrics,association=association,matched=matched,prediction_sha256=prediction_hashes,export=export,source_lock=lock)
    write(out/'PACKAGED_RESULTS.json',result)
    print(json.dumps(dict(result=str(out/'PACKAGED_RESULTS.json'),AP={k:v['AP'] for k,v in metrics.items()})))

def main():
    global DATA
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','train','train-all','evaluate'])
    p.add_argument('--seed',type=int,choices=[42,3407,2026],default=42)
    p.add_argument('--variant',choices=['base','flat','is'],default='is')
    p.add_argument('--output',help='New output directory; training uses it as run root')
    p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint');p.add_argument('--initializer')
    p.add_argument('--data-root',default=str(DATA),help='External prepared C-CCM data directory')
    from package_paths import RESOURCES
    p.add_argument('--pretrained',default=RESOURCES['coco_pretrained'],help='External COCO initializer for base training')
    p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    if args.mode!='check' and not args.output:p.error('--output is required for training/evaluation; preserve packaged results')
    import package_paths
    DATA=Path(args.data_root).resolve();package_paths.DATA=DATA
    m,protocol,qa=check_inputs()
    if args.mode=='check':
        for seed in [42,3407,2026]:
            for variant in ['base','flat','is']:local_config(variant,seed)
        print(json.dumps(dict(passed=True,counts=qa['counts'],source_checks=len(qa['source_and_input_hash_checks']),local_configs=9)));return
    if args.mode in ['train','train-all']:
        out=resolve(args.output);out.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(out).free<6*1024**3:raise ValueError('Select a training output drive with at least 6GiB free')
        for seed in ([42,3407,2026] if args.mode=='train-all' else [args.seed]):
            for variant in (['base','flat','is'] if args.mode=='train-all' else [args.variant]):train_one(args,variant,seed,out)
    else:evaluate(args,m,protocol)
if __name__=='__main__':main()
