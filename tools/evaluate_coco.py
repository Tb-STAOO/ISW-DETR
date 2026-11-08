"""COCO bbox evaluation of saved inference JSONs; never feeds labels to a model."""
from pathlib import Path
import argparse,json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path,help='COCO-format prediction JSON')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(str(args.annotations));dt=gt.loadRes(str(args.predictions))
    e=COCOeval(gt,dt,'bbox');e.evaluate();e.accumulate();e.summarize()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(AP=float(e.stats[0])*100,COCO_stats=e.stats.tolist()),indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
