"""Code-package locations and explicitly external data/weight resources."""
from pathlib import Path
import hashlib,json,os,yaml
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT;RT=ROOT;NATIVE=ROOT/'runtime';RUN=ROOT/'configs/cccm'
RESOURCES=json.loads((ROOT/'resources.example.json').read_text(encoding='utf8'))
DATA=Path(RESOURCES['public_data_root'])

def relocated(value):
    if isinstance(value,dict):return {k:relocated(v) for k,v in value.items()}
    if isinstance(value,list):return [relocated(v) for v in value]
    if isinstance(value,str):
        s=value.replace('\\','/')
        mappings=[
            ('D:/MY_DATA/02-project/sci_project/RT_DETR/Official/RT-DETR/rtdetrv2_pytorch',NATIVE),
            ('E:/ISW_DETR_external_data/CCCM_full_20261024',DATA),
        ]
        for old,new in mappings:
            if s.lower()==old.lower():return new.as_posix()
            if s.lower().startswith(old.lower()+'/'):return (new/s[len(old)+1:]).as_posix()
    return value

def local_config(variant,seed,output_root=None):
    cfg=relocated(yaml.safe_load((RUN/f'{variant}_seed{seed}.yml').read_text(encoding='utf8')))
    if output_root is not None:cfg['output_dir']=(Path(output_root).resolve()/f'{variant}_seed{seed}').as_posix()
    cfg.setdefault('PResNet',{})['pretrained']=False
    content=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
    fingerprint=hashlib.sha256(content.encode('utf8')).hexdigest()[:16]
    cache=Path(os.environ.get('ISW_CONFIG_CACHE',str(ROOT/'generated_configs'))).expanduser().resolve()
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{variant}_seed{seed}_{fingerprint}.yml'
    if not path.exists():
        temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        temporary.write_text(content,encoding='utf8')
        temporary.replace(path)
    return path

def resolve(path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
