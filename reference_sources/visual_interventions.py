"""Functions preserved from the original experiment implementation."""
import numpy as np
import torch
from transforms import transform, groups_for
NOISE_SEED=2026102201

def perturbations(data,p,mapping):
    pairs=[ids.tolist() for ids in groups_for(mapping) if len(ids)==2]
    shift=torch.zeros_like(p['z']);common=torch.zeros_like(shift)
    for i,item in enumerate(data['prepared']):
        if item is None:continue
        rng=np.random.default_rng(NOISE_SEED+1000003*int(data['arrays']['image_ids'][i]))
        signs=rng.choice([-1.,1.],size=(data['arrays']['logits'].shape[1],len(pairs)))
        ids=np.asarray(item['candidates']);signs=torch.tensor(signs[ids],dtype=torch.float32)
        for pair,(a,b) in enumerate(pairs):
            shift[i,:len(ids),a]=signs[:,pair];shift[i,:len(ids),b]=-signs[:,pair]
            common[i,:len(ids),a]=signs[:,pair];common[i,:len(ids),b]=signs[:,pair]
    assert torch.equal(shift.square().sum(-1),common.square().sum(-1))
    z=p['z'];mean=transform(z,mapping,'mean_erase');contrast=z-mean
    return dict(normal=(z,1.),contrast_delta1=(z+shift,1.),contrast_delta2=(z+2*shift,1.),
        contrast_delta4=(z+4*shift,1.),state_scale0=(mean,1.),
        state_scale2=(mean+2*contrast,1.),state_scale_minus1=(mean-contrast,1.),
        pair_common_delta2=(z+2*common,1.),likelihood_temperature0=(z,0.),
        **{'likelihood_temperature0.5':(z,.5)})

