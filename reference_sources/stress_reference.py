"""Functions preserved from the original experiment implementation."""
import torch
import state_controls as probe
import train_benefit_router as tr
from transforms import transform
METHODS=("original_fixed_CVSR","original_BQR","half_readout_original_assignment","half_readout_mean_assignment","half_readout_max_assignment","full_mean_erasure")

@torch.no_grad()
def with_visual_and_evidence(p,z,temperature=1.):
    p={k:v.clone() for k,v in p.items()}
    # Legacy caches reconstruct this *router feature* from rounded output
    # logits. Keep its original encoding offset across all interventions;
    # class logits always use the actual current hard@event residual below.
    p['legacy_qres_feature_offset']=p['query_record_residual']-torch.bmm(p['hard_p'],p['event_residual'])
    p['z']=torch.where(p['candidate_mask'][...,None],z,p['z'])
    if temperature!=1.:
        # Likelihood-temperature ablation, not a measured sensor-noise model.
        p['logp']=torch.log_softmax(temperature*p['logp'],dim=-1)
        for i in range(len(z)):
            e=int(p['weight_mask'][i].sum())
            if not e:continue
            probability=p['logp'][i,:e].exp().amax(0)
            log_probability=probability.clamp_min(1e-8).log()
            p['scene'][i]=(log_probability+torch.tensor(z.shape[-1],dtype=z.dtype).log()).clamp(-6,6)
            p['event_residual'][i,:e]=(p['logp'][i,:e]-log_probability).clamp(-6,0)
    return p


@torch.no_grad()
def infer(data,p,mapping,method,router=None,scale=.5):
    """No matched labels, owners or teacher fields are consulted."""
    if method not in METHODS:raise ValueError(method)
    result={k:v.clone() for k,v in p.items()}
    source=p['z'];mean=transform(source,mapping,'mean_erase')
    if method in ('half_readout_mean_assignment','full_mean_erasure'):match=mean
    elif method=='half_readout_max_assignment':match=transform(source,mapping,'max_erase')
    else:match=source
    utilities=torch.zeros_like(p['hard_p'])
    actual_residual=torch.zeros_like(source)
    for i in range(len(source)):
        k=int(p['candidate_mask'][i].sum());e=int(p['weight_mask'][i].sum())
        if not bool(p['transaction_gate'][i]) or not k or not e:continue
        utility=torch.logsumexp((match[i,:k]+2*p['scene'][i])[:,None]+p['event_residual'][i,:e][None],dim=-1)
        utilities[i,:k,:e]=utility
        hard,row,col=tr.predicted_assignment(match[i,:k],p['scene'][i],p['event_residual'][i,:e])
        result['hard_p'][i,:k,:e]=hard;result['row_margin'][i,:k]=row;result['col_margin'][i,:k]=col
        actual_residual[i,:k]=hard@p['event_residual'][i,:e]
    result['actual_query_record_residual']=actual_residual
    result['query_record_residual']=actual_residual
    offset=p.get('legacy_qres_feature_offset')
    if offset is None:offset=p['query_record_residual']-torch.bmm(p['hard_p'],p['event_residual'])
    if method=='original_BQR':
        result['query_record_residual']=actual_residual+offset
    physical=2*p['scene'][:,None]+.2*actual_residual
    if method in ('original_fixed_CVSR','original_BQR'):
        result['base']=source+2*p['scene'][:,None]+.2*actual_residual
    elif method=='full_mean_erasure':result['base']=mean+2*p['scene'][:,None]+.2*actual_residual
    elif method=='half_readout_original_assignment':
        # This comparator's locked transfer export explicitly consumed the
        # legacy reconstructed residual, unlike recomputed mean/max matching.
        # Preserve that fixed encoding offset for its entire stress curve.
        result['base']=source-.5*(source-mean)+(2*p['scene'][:,None]+.2*(actual_residual+offset))
    else:result['base']=source-.5*(source-mean)+physical
    active=p['candidate_mask']&p['transaction_gate'][:,None]
    result['base']=torch.where(active[...,None],result['base'],source)
    if method=='original_BQR':
        assert router is not None
        logits=tr.replay(router,data,result,'cpu',scale=scale)[0]
    else:logits=probe.fixed_logits(data,result)
    return logits,result,utilities

