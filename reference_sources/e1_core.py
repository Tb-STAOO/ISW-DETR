"""Functions preserved from the original experiment implementation."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import train_benefit_router as tr
from benefit_router import BenefitRouter, extract_router_features

class ConfidencePredictionRouter(BenefitRouter):
    def __init__(self,num_classes=28,hidden_dim=32,feature_mode='full',gate_upper=2.):
        super().__init__(num_classes,hidden_dim,feature_mode,gate_upper)
        self.mlp[-1]=nn.Linear(max(2,hidden_dim//2),2)
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
    def forward(self,visual_logits,base_logits,candidate_mask,transaction_gate,
                constant_gate=None,return_details=False,**kwargs):
        features=extract_router_features(visual_logits,base_logits,candidate_mask,transaction_gate,
                                        feature_mode=self.feature_mode,**kwargs)
        confidence=torch.sigmoid(self.mlp(features))
        learned=2*confidence[...,1]/confidence.sum(-1).clamp_min(1e-8)
        if constant_gate is not None:
            assert 0<=constant_gate<=2
            learned=torch.full_like(learned,float(constant_gate))
        active=candidate_mask.bool()&transaction_gate.bool()[:,None]
        if (active&~torch.isfinite(visual_logits).all(-1)).any():raise ValueError('Nonfinite visual')
        active &= torch.isfinite(base_logits).all(-1)
        routed=base_logits+(learned[...,None]-1)*(base_logits-visual_logits)
        if constant_gate==0:routed=visual_logits
        if constant_gate==1:routed=base_logits
        logits=torch.where(active[...,None],routed,visual_logits)
        if return_details:return dict(pred_logits=logits,gate=torch.where(active,learned,torch.zeros_like(learned)),
                                     features=features,active_mask=active,confidence=confidence)
        return logits


@torch.no_grad()
def map_logits(data,alpha):
    logits=tr.tensor(data['arrays']['logits'],torch.float32).clone()
    diagnostics=[]
    for i,p in enumerate(data['prepared']):
        if p is None:
            diagnostics.append(dict(index=i,gate=False));continue
        z=tr.tensor(p['z'],torch.float32);lp=tr.tensor(p['logp'],torch.float32)
        scene=tr.tensor(p['scene'],torch.float32)
        utility=(z.log_softmax(-1)[:,None]+float(alpha)*lp[None]).max(-1).values
        rows,cols=linear_sum_assignment(-utility.numpy())
        current=z+2*scene
        record=(lp+np.log(z.shape[-1])).clamp(-6,6)
        current[rows]+=float(alpha)*record[cols]
        indices=tr.tensor(p['candidates'],torch.long)
        logits[i,indices]=current
        diagnostics.append(dict(index=i,gate=True,candidates=indices.tolist(),assigned_local_rows=rows.tolist(),
                                assigned_event_columns=cols.tolist()))
    assert torch.isfinite(logits).all()
    return logits,diagnostics

