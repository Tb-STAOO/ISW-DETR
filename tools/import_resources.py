"""Import named external archives using author-provided fixed SHA-256 digests."""
import argparse,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=json.loads((ROOT/'RESOURCE_ARCHIVES.json').read_text(encoding='utf8'))
    groups={r['group']:r for r in spec['resource_archives']}
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--groups',nargs='+',choices=sorted(groups),default=sorted(groups))
    args=p.parse_args();destination=Path(args.output).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    for name in args.groups:
        entry=groups[name];archive=Path(args.archives).resolve()/entry['file']
        output=destination/name.replace('-','_')
        result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/resource_archive.py'),'import',
            '--archive',str(archive),'--sha256',entry['sha256'],'--output',str(output)],check=False)
        if result.returncode:raise RuntimeError('Resource import failed: '+name)


if __name__=='__main__':main()
