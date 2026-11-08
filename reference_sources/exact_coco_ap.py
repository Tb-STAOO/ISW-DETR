"""Functions preserved from the original experiment implementation."""
import contextlib, io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
RECALL=np.linspace(0,1,101)

def coco_eval(dataset,records):
 with contextlib.redirect_stdout(io.StringIO()):
  gt=COCO();gt.dataset=dataset;gt.createIndex();dt=gt.loadRes(records)
  ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
 return ev


def extract(ev):
 n=len(ev.params.imgIds);a=len(ev.params.areaRng);cats=[]
 for k in range(len(ev.params.catIds)):
  records=[]
  for i in range(n):
   item=ev.evalImgs[k*a*n+i]
   if item is None: r={'scores':np.zeros(0),'matched':np.zeros((10,0),bool),'ignored':np.zeros((10,0),bool),'positives':0}
   else:r={'scores':np.asarray(item['dtScores']),'matched':item['dtMatches']>0,'ignored':np.asarray(item['dtIgnore'],bool),'positives':int(np.count_nonzero(np.asarray(item['gtIgnore'])==0))}
   records.append(r)
  cats.append(records)
 return cats


def ap(cats,sampled):
 values=[]
 for records in cats:
  chosen=[records[int(i)] for i in sampled];npos=sum(r['positives'] for r in chosen)
  if npos==0:continue
  scores=np.concatenate([r['scores'] for r in chosen])
  if not scores.size:values.append(0.);continue
  order=np.argsort(-scores,kind='mergesort')
  mat=np.concatenate([r['matched'] for r in chosen],1)[:,order]
  ign=np.concatenate([r['ignored'] for r in chosen],1)[:,order]
  tp=np.cumsum(mat&~ign,axis=1,dtype=float);fp=np.cumsum(~mat&~ign,axis=1,dtype=float)
  recall=tp/npos;precision=tp/(tp+fp+np.spacing(1))
  precision=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
  v=np.zeros((10,101))
  for t in range(10):
   inds=np.searchsorted(recall[t],RECALL,side='left');ok=inds<scores.size
   v[t,ok]=precision[t,inds[ok]]
  values.append(float(v.mean()))
 return float(np.mean(values))


def duplicate(dataset,records,ids,sampled):
 images={int(x['id']):x for x in dataset['images']}
 annotations={i:[] for i in ids};preds={i:[] for i in ids}
 for a in dataset['annotations']:annotations[int(a['image_id'])].append(a)
 for p in records:preds[int(p['image_id'])].append(p)
 data={'info':{},'categories':dataset['categories'],'images':[],'annotations':[]};dt=[]
 for new,index in enumerate(sampled,1):
  old=ids[int(index)];data['images'].append({**images[old],'id':new})
  for a in annotations[old]:data['annotations'].append({**a,'image_id':new,'id':len(data['annotations'])+1})
  dt.extend({**p,'image_id':new} for p in preds[old])
 return data,dt

