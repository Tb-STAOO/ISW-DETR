"""Functions preserved from the original experiment implementation."""
import math
import torch
import train_benefit_router as tr
from transforms import groups_for

def alternatives(z,mapping):
    penalty=torch.zeros_like(z);lse_erased=z.clone();half_lse=z.clone()
    for ids in groups_for(mapping):
        if len(ids)==1:continue
        values=z[...,ids];mu=values.mean(-1,keepdim=True)
        lse=torch.logsumexp(values,-1,keepdim=True)
        penalty[...,ids]=(lse-math.log(len(ids))-mu).expand_as(values)
        lse_erased[...,ids]=(lse-math.log(len(ids))).expand_as(values)
        half=mu+.5*(values-mu)
        half_lse[...,ids]=half+(lse-torch.logsumexp(half,-1,keepdim=True))
        assert torch.allclose(torch.logsumexp(half_lse[...,ids],-1),lse.squeeze(-1),atol=2e-6,rtol=0)
    common_penalty=z-penalty
    margin=z.topk(2,dim=-1).values.diff(dim=-1).abs()
    result=dict(common_penalty_reassign=common_penalty,common_penalty_fixed=common_penalty,
        lse_erasure_fixed=lse_erased,half_lse_preserving_reassign=half_lse)
    for beta in (.5,.75,1.25,1.5):result[f'global_scale_{beta:g}']=z*beta
    for beta in (.5,1.):result[f'query_common_margin_penalty_{beta:g}']=z-beta*margin
    return result


@torch.no_grad()
def pack_with_z(p,z,reassign):
    changed={k:v.clone() for k,v in p.items()}
    changed['z']=torch.where(p['candidate_mask'][...,None],z,p['z'])
    if not reassign:
        changed['base']=changed['z']+(p['base']-p['z'])
        return changed
    for i in range(len(z)):
        k=int(p['candidate_mask'][i].sum());e=int(p['weight_mask'][i].sum())
        if not k or not e:continue
        hard,row,col=tr.predicted_assignment(z[i,:k],p['scene'][i],p['event_residual'][i,:e])
        residual=hard@p['event_residual'][i,:e]
        changed['hard_p'][i,:k,:e]=hard;changed['query_record_residual'][i,:k]=residual
        changed['row_margin'][i,:k]=row;changed['col_margin'][i,:k]=col
        changed['base'][i,:k]=z[i,:k]+2*p['scene'][i]+.2*residual
    return changed

