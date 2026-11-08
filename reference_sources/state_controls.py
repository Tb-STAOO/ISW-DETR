"""Functions preserved from the original experiment implementation."""
import torch
import train_benefit_router as tr
from transforms import transform, groups_for

@torch.no_grad()
def changed_pack(packed,mapping,method):
    result={name:value.clone() for name,value in packed.items()}
    if method=='original':return result
    z=transform(packed['z'],mapping,method)
    result['z']=torch.where(packed['candidate_mask'][:,:,None],z,packed['z'])
    for i in range(len(z)):
        k=int(packed['candidate_mask'][i].sum());e=int(packed['weight_mask'][i].sum())
        if not k or not e:continue
        hard,row,col=tr.predicted_assignment(result['z'][i,:k],packed['scene'][i],packed['event_residual'][i,:e])
        qres=hard@packed['event_residual'][i,:e]
        result['hard_p'][i,:k,:e]=hard;result['query_record_residual'][i,:k]=qres
        result['row_margin'][i,:k]=row;result['col_margin'][i,:k]=col
        result['base'][i,:k]=result['z'][i,:k]+2*packed['scene'][i]+.2*qres
    return result


def fixed_logits(data,packed):
    logits=tr.tensor(data['arrays']['logits'],torch.float32).clone()
    for i,p in enumerate(data['prepared']):
        if p is not None:logits[i,tr.tensor(p['candidates'],torch.long)]=packed['base'][i,:len(p['z'])]
    return logits

