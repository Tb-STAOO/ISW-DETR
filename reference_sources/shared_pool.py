"""Original research functions; implementation preserved, portable caller supplies paths."""
import torch
import train_benefit_router as tr
from transforms import transform

@torch.no_grad()
def prediction(data,pool_data,mapping,rule):
    full=tr.tensor(data['arrays']['logits'],torch.float32).clone()
    for i,(original,pool) in enumerate(zip(data['prepared'],pool_data['prepared'])):
        if pool is None:continue
        assert original is not None
        ids=tr.tensor(pool['candidates'],torch.long)
        z=tr.tensor(data['arrays']['logits'][i],torch.float32)[ids]
        scene=tr.tensor(original['scene'],torch.float32);r=tr.tensor(original['residual'],torch.float32)
        if rule=='fixed' and data is pool_data:
            full[i,ids]=tr.tensor(original['base'],torch.float32)
        elif rule=='new':
            mean=transform(z,mapping,'mean_erase')
            hard,_,_=tr.predicted_assignment(mean,scene,r)
            full[i,ids]=z-.5*(z-mean)+(2*scene+.2*(hard@r))
        else:raise ValueError(rule)
    return full


