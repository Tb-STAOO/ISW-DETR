"""Native-image research wrapper for the validation-selected BenefitRouter.

The frozen native detector is untouched. This wrapper reads only its predicted
CVSR diagnostics and observed transaction fields, then replaces class logits.
"""
from pathlib import Path
import math
import torch
from torch import nn
from benefit_router import BenefitRouter

def load_router(path,seed=None):
 payload=torch.load(Path(path),map_location='cpu',weights_only=True)
 if payload.get('class_name')!='BenefitRouter': raise ValueError('Expected BenefitRouter checkpoint')
 selected=payload.get('selection',{})
 if selected.get('split')!='validation': raise ValueError('Checkpoint must be validation-selected')
 if seed is not None and int(selected.get('seed',-1))!=int(seed): raise ValueError('Seed mismatch')
 config=payload['config']
 if config.get('feature_mode')!='full': raise ValueError('Research wrapper requires full mass feature checkpoint')
 scale=float(payload['inference_scale'])
 if not math.isfinite(scale) or not 0<=scale<=1: raise ValueError('Invalid inference scale')
 router=BenefitRouter(**config)
 router.load_state_dict(payload['state_dict'],strict=True)
 return router.eval(),scale,payload

class BenefitRoutedDetector(nn.Module):
 def __init__(self,detector,checkpoint_path,seed=None):
  super().__init__()
  p=detector.cvsr
  if p is None or p.require_positive_mutual_margin or p.scene_beta!=2 or p.residual_strength!=.2:
   raise ValueError('Requires fixed no-positive-margin beta=2 eta=.2 CVSR parent')
  if p.ambiguity_adapter is not None or p.mass_count_refiner is not None:
   raise ValueError('Do not stack historical research adapters with this parent')
  self.detector=detector
  self.router,self.inference_scale,self.checkpoint_metadata=load_router(checkpoint_path,seed)
  self.router.to(next(detector.parameters()).device)
  self.requires_grad_(False)
  self.eval()

 def forward(self,x,targets=None,modal_inputs=None,visual_inputs=None):
  if self.training: raise ValueError('Frozen research wrapper is inference-only')
  if targets is not None: raise ValueError('Ground truth is not an inference input')
  result=self.detector(x,targets=None,modal_inputs=modal_inputs,visual_inputs=visual_inputs)
  base=result['pred_logits']
  visual=result['pred_logits_cfqd']
  gate=result['cvsr_integrity_gate']
  candidate=result['cvsr_candidate_mask']
  result['pred_logits_benefit_parent']=base
  if modal_inputs is None or modal_inputs.get('weight_tokens_raw') is None or not bool(gate.any()):
   result['benefit_gate']=torch.zeros_like(candidate,dtype=visual.dtype)
   return result
  if self.inference_scale==0:
   result['benefit_gate']=candidate.to(visual.dtype)
   return result
  raw=modal_inputs['weight_tokens_raw'].to(visual.device,dtype=torch.float32)
  weight_mask=modal_inputs.get('weight_mask',torch.ones(raw.shape[:2],device=raw.device,dtype=torch.bool)).bool()
  physics=self.detector.cvsr
  logp,probability,scene,_=physics.mass_evidence(raw,weight_mask,modal_inputs)
  residual=(logp-probability.clamp_min(1e-8).log()[:,None]).clamp(-6,0)
  b,q,c=visual.shape;e=raw.shape[1]
  hard=visual.new_zeros((b,q,e))
  qres=visual.new_zeros((b,q,c))
  row_margin=visual.new_zeros((b,q))
  col_margin=visual.new_zeros((b,q))
  assigned=result['cvsr_assigned_event']
  for bi in range(b):
   if not bool(gate[bi]): continue
   queries=candidate[bi].nonzero().flatten()
   events=weight_mask[bi].nonzero().flatten()
   utility=torch.logsumexp(result['pred_logits_scene'][bi,queries,None]+residual[bi,None,events],-1)
   reverse={int(event):j for j,event in enumerate(events.tolist())}
   for local,query in enumerate(queries.tolist()):
    event=int(assigned[bi,query])
    if event<0: continue
    hard[bi,query,event]=1
    qres[bi,query]=residual[bi,event]
    ei=reverse[event];value=utility[local,ei]
    if len(events)>1:
     alt=utility[local].clone();alt[ei]=-torch.inf
     row_margin[bi,query]=value-alt.max()
    if len(queries)>1:
     alt=utility[:,ei].clone();alt[local]=-torch.inf
     col_margin[bi,query]=value-alt.max()
  routed=self.router(visual,base,candidate,gate,scene_evidence=scene,
      event_log_probability=logp,weight_tokens_raw=raw,weight_mask=weight_mask,
      query_record_residual=qres,hard_p=hard,row_margin=row_margin,col_margin=col_margin,
      return_details=True)
  effective=base+self.inference_scale*(routed['pred_logits']-base)
  active=candidate&gate[:,None]
  result['pred_logits']=torch.where(active[:,:,None],effective,visual)
  result['benefit_gate']=torch.where(active,1+self.inference_scale*(routed['gate']-1),torch.zeros_like(routed['gate']))
  return result

