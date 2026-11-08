"""Original research functions; implementation preserved, portable caller supplies paths."""
from pathlib import Path
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
import json,math,statistics,hashlib
SEEDS=(42,3407,2026)
def read(path):return json.loads(Path(path).read_text(encoding="utf8"))

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stats(values):
    values=list(map(float,values))
    if len(values)!=len(SEEDS) or not all(math.isfinite(v) for v in values):
        raise ValueError('Exactly three finite seed values are required; no missing-seed averaging')
    return dict(n=3,mean=statistics.mean(values),sample_sd=statistics.stdev(values),
        minimum=min(values),maximum=max(values),values=values,seeds=list(SEEDS))


def iter_prediction_array(path,chunk_size=1024*1024):
    """Read exported compact prediction arrays without materializing all frames."""
    decoder=json.JSONDecoder();buffer='';position=0;eof=False;started=False;expect_value=True;after_comma=False
    with Path(path).open('r',encoding='utf8') as stream:
        def more():
            nonlocal buffer,position,eof
            block=stream.read(chunk_size)
            buffer=buffer[position:]+block;position=0;eof=not block
            if len(buffer)>8*1024*1024:
                raise ValueError(f'Overlarge/malformed individual prediction object: {path}')
        while True:
            while position<len(buffer) and buffer[position].isspace():position+=1
            if not started:
                if position==len(buffer) and not eof:more();continue
                if position==len(buffer) or buffer[position]!='[':raise ValueError(f'Prediction file is not a JSON array: {path}')
                started=True;position+=1;continue
            if position==len(buffer):
                if eof:raise ValueError(f'Truncated prediction array: {path}')
                more();continue
            token=buffer[position]
            if expect_value and token==']':
                if after_comma:raise ValueError(f'Trailing JSON array comma: {path}')
                if buffer[position+1:].strip() or stream.read().strip():raise ValueError(f'Trailing JSON data: {path}')
                return
            if not expect_value:
                if token==']':
                    if buffer[position+1:].strip() or stream.read().strip():raise ValueError(f'Trailing JSON data: {path}')
                    return
                if token!=',':raise ValueError(f'Missing JSON array comma: {path}')
                position+=1;expect_value=True;after_comma=True;continue
            try:
                value,at=decoder.raw_decode(buffer,position)
            except json.JSONDecodeError:
                if eof:raise ValueError(f'Truncated or malformed prediction item: {path}')
                more();continue
            if not isinstance(value,dict):raise ValueError(f'Non-object prediction item: {path}')
            yield value
            position=at;expect_value=False;after_comma=False


def subset_metrics(annotation,prediction_path,ids,log_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids=set(ids);predictions=[];top={}
    for row in iter_prediction_array(prediction_path):
        iid=int(row['image_id'])
        if iid not in ids:continue
        label=int(row['category_id']);score=float(row['score']);box=row['bbox']
        if not 0<=label<42 or not math.isfinite(score) or len(box)!=4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError(f'Invalid exported prediction: {prediction_path}')
        predictions.append(row)
        if iid not in top or score>float(top[iid]['score']):top[iid]=row
    ann=read(annotation);truth=defaultdict(list)
    for row in ann['annotations']:
        if int(row['image_id']) in ids:truth[int(row['image_id'])].append(row)
    if set(truth)!=ids or any(len(v)!=1 for v in truth.values()):
        raise ValueError('Sensitivity diagnostic expects all included images to have one native GT')
    fine=state=joint=0
    for iid,targets in truth.items():
        guess=top.get(iid)
        if guess is None:continue
        target=targets[0];pred=int(guess['category_id']);label=int(target['category_id'])
        fine+=pred==label;state+=pred%7==label%7
        a,b=list(map(float,guess['bbox'])),list(map(float,target['bbox']))
        w=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
        h=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
        intersection=w*h;union=max(0.,a[2])*max(0.,a[3])+b[2]*b[3]-intersection
        iou=intersection/union if union>0 else 0.
        joint+=pred==label and iou>=.5
    captured=StringIO()
    with redirect_stdout(captured):
        coco=COCO(str(annotation))
        if predictions:
            detections=coco.loadRes(predictions)
        else:
            detections=COCO();detections.dataset=dict(images=coco.dataset['images'],categories=coco.dataset['categories'],annotations=[])
            detections.createIndex()
        evaluator=COCOeval(coco,detections,'bbox');evaluator.params.imgIds=sorted(ids)
        evaluator.params.catIds=sorted(int(c['id']) for c in ann['categories'])
        evaluator.params.maxDets=[1,10,100]
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    Path(log_path).write_text(captured.getvalue(),encoding='utf8')
    return dict(AP=100*float(evaluator.stats[0]),AP50=100*float(evaluator.stats[1]),
        AP75=100*float(evaluator.stats[2]),images=len(ids),predictions=len(predictions),
        top1_fine_accuracy_pct=100*fine/len(ids),top1_state_accuracy_pct=100*state/len(ids),
        top1_fine_and_iou50_accuracy_pct=100*joint/len(ids),
        AP_definition='COCO IoU .50:.95, maxDets=100, original full 42-class taxonomy; imgIds restricted',
        missing_top1_images=len(ids)-len(top))


