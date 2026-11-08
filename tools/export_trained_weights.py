"""Export validation-selected full private training outputs without altering bytes."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.training_root).resolve();out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh export directory')
    rows=[]
    for seed in [42,3407,2026]:
        folder=source/f'is_seed{seed}';receipt=json.loads((folder/'TRAINING_RESULT.json').read_text(encoding='utf8'))
        path=folder/'best.pth'
        if receipt['status']!='complete' or receipt['smoke_only'] or receipt['seed']!=seed or sha(path)!=receipt['checkpoint_sha256']:
            raise ValueError('Incomplete, diagnostic or changed training checkpoint: '+str(path))
        rows.append(dict(seed=seed,source=str(path),source_sha256=sha(path),selected_epoch=receipt['best_epoch']))
    out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        destination=out/f"detector_seed{row['seed']}.pth";shutil.copy2(row['source'],destination)
        if sha(destination)!=row['source_sha256']:raise ValueError('Weight export changed bytes')
    (out/'TRAINED_WEIGHT_EXPORT.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='complete',weights=3,output=str(out))))

if __name__=='__main__':main()
