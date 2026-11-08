"""Disentangle state erasure from group-common confidence shifts.

These are controlled inference interventions, not newly claimed algorithms.
All inputs are predicted logits and a fixed, author-specified class taxonomy.
"""
import math
import numpy as np
import torch

METHODS=('original','mean_erase','common_boost','max_erase','lse_erase',
         'shuffled_max_erase','shuffled_mean_erase')
SHUFFLE_SEED=2026102101


def groups_for(mapping):
    mapping=torch.as_tensor(mapping,dtype=torch.long)
    return [(mapping==parent).nonzero().flatten() for parent in mapping.unique(sorted=True)]


def shuffled_mapping(mapping):
    mapping=torch.as_tensor(mapping,dtype=torch.long)
    permutation=torch.tensor(np.random.default_rng(SHUFFLE_SEED).permutation(len(mapping)))
    return mapping[permutation]


def transform(logits,mapping,method):
    if method not in METHODS:raise ValueError(method)
    if method=='original':return logits.clone()
    if method.startswith('shuffled_'):
        mapping=shuffled_mapping(mapping);method=method.removeprefix('shuffled_')
    result=logits.clone()
    for ids in groups_for(mapping):
        if len(ids)==1:continue
        values=logits[...,ids]
        mean=values.mean(-1,keepdim=True)
        maximum=values.amax(-1,keepdim=True)
        if method=='mean_erase':value=mean.expand_as(values)
        elif method=='common_boost':value=values+(maximum-mean)
        elif method=='max_erase':value=maximum.expand_as(values)
        elif method=='lse_erase':
            value=(torch.logsumexp(values,-1,keepdim=True)-math.log(len(ids))).expand_as(values)
        else:raise ValueError(method)
        result[...,ids]=value
    return result
