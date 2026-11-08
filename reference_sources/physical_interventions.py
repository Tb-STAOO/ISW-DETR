"""Functions preserved from the original experiment implementation."""
import copy
import numpy as np
import torch
import train_benefit_router as tr
from data_export import modal_from_record
NOISE_SEED=2026102202

def platform_offset(record,sigma,noise_seed=NOISE_SEED):
    """Shift shared interior plateaus without repairing original record errors.

    This function has no access to object labels, detector outputs, or predictions.
    The same deterministic image-level draw is reused at every sigma and seed.
    """
    if not np.isfinite(sigma) or sigma<0:raise ValueError('sigma must be finite and nonnegative')
    result=copy.deepcopy(record)
    raw=np.asarray(record['weight_tokens_raw'],dtype=np.float64)
    mask=np.asarray(record['weight_mask'],dtype=bool)
    order=np.asarray(record['event_order'],dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=7 or mask.shape!=(len(raw),) or order.shape!=mask.shape:
        raise ValueError('unaligned raw/mask/order fields')
    valid=np.flatnonzero(mask)
    values=order[valid]
    if not np.isfinite(values).all() or not np.equal(values,np.round(values)).all() or len(np.unique(values))!=len(values):
        raise ValueError('event order must contain unique finite integers')
    ordered=valid[np.argsort(values,kind='stable')]
    e=len(ordered)
    rng=np.random.default_rng(noise_seed+1000003*int(record['image_id']))
    epsilon=np.zeros(e+1,dtype=np.float64)
    if e>1:epsilon[1:e]=sigma*rng.standard_normal(e-1)
    if sigma!=0 and e>1:
        changed=raw.copy()
        changed[ordered,1]+=epsilon[:-1]
        changed[ordered,2]+=epsilon[1:]
        changed[ordered,0]+=epsilon[1:]-epsilon[:-1]
        result['weight_tokens_raw']=changed.tolist()
    return result,dict(image_id=int(record['image_id']),sigma_g=float(sigma),
        ordered_original_rows=ordered.tolist(),event_count=e,
        interior_plateau_offsets_g=epsilon.tolist(),
        identity_for_one_or_zero_event=e<=1)


@torch.no_grad()
def evidence_pack(data,p,module,sigma):
    """Recompute physical evidence from raw records, retaining the old query pool."""
    result={key:value.clone() for key,value in p.items()}
    # Explicit stable interface: do not infer a historical offset from the
    # physically perturbed event residuals inside the shared replay core.
    historical_offset=p.get('legacy_qres_feature_offset')
    if historical_offset is None:
        historical_offset=p['query_record_residual']-torch.bmm(p['hard_p'],p['event_residual'])
    result['legacy_qres_feature_offset']=historical_offset.clone()
    logs=[]
    for i,record in enumerate(data['records']):
        assert int(record['image_id'])==int(data['arrays']['image_ids'][i])
        modified,metadata=platform_offset(record,sigma)
        modal=modal_from_record(modified)
        lp,probability,scene,integrity=module.mass_evidence(
            modal['weight_tokens_raw'],modal['weight_mask'],modal)
        valid=modal['weight_mask'][0].nonzero().flatten()
        e=len(valid)
        item=data['prepared'][i]
        if item is None:
            assert not bool(integrity['gate'][0]),'previously unprepared row became accepted'
            result['transaction_gate'][i]=False
        else:
            assert torch.equal(valid,tr.tensor(item['valid'],torch.long))
            assert e==int(p['weight_mask'][i].sum())
            result['logp'][i,:e]=lp[0,valid]
            result['raw'][i,:e]=modal['weight_tokens_raw'][0,valid]
            result['scene'][i]=scene[0]
            residual=(lp[0,valid]-probability[0].clamp_min(1e-8).log()).clamp(-6,0)
            result['event_residual'][i,:e]=residual
            result['transaction_gate'][i]=integrity['gate'][0]
            # The old BQR cache has an additive roundoff offset in its feature
            # residual. Keep that offset rather than encoding the new mass
            # change itself as if it were historical numerical roundoff.
            k=int(p['candidate_mask'][i].sum())
            offset=historical_offset[i,:k]
            new_actual=p['hard_p'][i,:k,:e]@residual
            result['query_record_residual'][i,:k]=new_actual+offset
            result['base'][i,:k]=p['z'][i,:k]+2*scene[0]+.2*new_actual
        metadata.update(gate=bool(integrity['gate'][0]),rejection_code=int(integrity['rejection_code'][0]),
            record_residual_g=float(integrity['record_residual_g'][0]),
            chain_residual_g=float(integrity['chain_residual_g'][0]),
            total_residual_g=float(integrity['total_residual_g'][0]),
            end_residual_g=float(integrity['end_residual_g'][0]))
        logs.append(metadata)
    assert torch.equal(result['candidate_mask'],p['candidate_mask'])
    assert torch.equal(result['weight_mask'],p['weight_mask'])
    assert torch.equal(result['z'],p['z'])
    return result,logs


from stress_reference import METHODS, infer

@torch.no_grad()
def fallback_contract(data,p,module,mapping,router,scale):
    """One invalid-record condition; no metric evaluation."""
    changed={key:value.clone() for key,value in p.items()}
    codes=[]
    for i,record in enumerate(data['records']):
        fault=copy.deepcopy(record)
        valid=np.flatnonzero(np.asarray(fault['weight_mask'],dtype=bool))
        if len(valid):fault['weight_tokens_raw'][int(valid[0])][0]+=10.
        modal=modal_from_record(fault)
        _,_,_,integrity=module.mass_evidence(modal['weight_tokens_raw'],modal['weight_mask'],modal)
        assert not bool(integrity['gate'][0])
        code=int(integrity['rejection_code'][0]);codes.append(code)
        if len(valid):assert code&module.REJECTION_BITS['record_inconsistent']
        changed['transaction_gate'][i]=False
    raw=tr.tensor(data['arrays']['logits'],torch.float32)
    checks={}
    for method in METHODS:
        value=infer(data,changed,mapping,method,router,scale)[0]
        checks[method]=torch.equal(value,raw)
        assert checks[method],f'fallback differs from input visual logits: {method}'
    return dict(scene_count=len(codes),rejected=len(codes),rejection_codes=codes,
        all300_bitwise_visual_return=checks,no_additional_ap=True)

