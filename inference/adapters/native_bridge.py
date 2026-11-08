"""Isolated raw-modal CVSR bridge; the native detector default is unchanged."""
from __future__ import annotations

import copy
import torch
from torch import nn

if __package__:
    from .refinement import StateNeutralRefinement
else:
    from refinement import StateNeutralRefinement


MODAL_KEYS=('weight_tokens_raw','weight_mask','event_order','scene_total_weight_g',
            'empty_baseline_g','scene_total_available','scene_total_independent')


class StateNeutralCVSRBridge(nn.Module):
    """Original raw integrity/evidence/pool, followed by one new association.

    ``upstream_cvsr`` supplies its original mass_evidence and _select_candidates
    methods. Its forward and _assign are never called. No prepared cache fields,
    annotations, labels or owners are consumed. Matching defaults to the actual
    old CARA score-order pool. query_order is an explicit numerical diagnostic;
    it preserves membership but can choose another tied Hungarian optimum.
    This is fusion integration over already produced visual tensors, not an RGB
    detector forward, an ONNX path or a measured speed improvement.
    """

    def __init__(self, upstream_cvsr, mapping, alpha=.5, *, matching_order='score_order'):
        super().__init__()
        if not isinstance(upstream_cvsr, nn.Module):
            raise TypeError('upstream_cvsr must be the original CVSR module.')
        if matching_order not in ('score_order','query_order'):
            raise ValueError('matching_order must be score_order or query_order.')
        self.upstream=upstream_cvsr
        self.refinement=StateNeutralRefinement(mapping,alpha)
        self.matching_order=matching_order
        self._check_configuration()

    def _check_configuration(self):
        u=self.upstream
        expected=dict(scene_beta=2.,residual_strength=.2,evidence_clip=6.,candidate_extra=12,
                      candidate_iou_threshold=.9,scene_candidate_only=True,
                      sensor_sigma_floor_g=.25,total_min_threshold_g=1.,chain_min_threshold_g=.5,
                      threshold_sigma_multiplier=1.,fixed_threshold_mode=True,
                      integrity_acceptance_ratio=1.,require_independent_snapshot=False,
                      require_positive_mutual_margin=False,assignment_strategy='hungarian',
                      bypass_integrity_gate=False)
        for name,value in expected.items():
            if getattr(u,name,None)!=value:
                raise ValueError(f'Unsupported upstream setting {name}: expected {value!r}.')
        if getattr(u,'num_classes',None)!=self.refinement.num_classes:
            raise ValueError('CVSR classes and the supplied taxonomy differ.')
        if getattr(u,'ambiguity_adapter',None) is not None or getattr(u,'mass_count_refiner',None) is not None:
            raise ValueError('Other ambiguity/count adapters are unsupported in this bridge.')
        for method in ('mass_evidence','_select_candidates','_empty_result','_integrity_check'):
            if not callable(getattr(u,method,None)):
                raise TypeError(f'Original CVSR method {method} is required.')
        if any(True for _ in u.parameters()):
            raise ValueError('The supported upstream CVSR must have no trainable/learned submodules.')

    def _empty(self, raw, boxes):
        b,q,c=raw.shape
        slots=torch.zeros(b,0,dtype=torch.bool,device=raw.device)
        residual=torch.zeros(b,0,c,dtype=torch.float32,device=raw.device)
        refined=self.refinement(raw,torch.empty(b,0,dtype=torch.long,device=raw.device),slots,
                                torch.zeros(b,c,device=raw.device),residual,slots,
                                torch.zeros(b,dtype=torch.bool,device=raw.device))
        output=self.upstream._empty_result(raw.float())
        output.update(refined)
        output.update(pred_logits_scene=raw.clone(),pred_boxes=boxes,
                      scene_evidence=torch.zeros(b,c,device=raw.device),
                      scene_probability=torch.full((b,c),1./c,device=raw.device),event_residual=residual,
                      candidate_indices=torch.empty(b,0,dtype=torch.long,device=raw.device),
                      candidate_slot_mask=slots)
        return output

    @torch.no_grad()
    def forward(self, visual_logits, boxes, modal_inputs):
        self._check_configuration()
        if not isinstance(visual_logits,torch.Tensor) or visual_logits.ndim!=3 or not visual_logits.is_floating_point():
            raise ValueError('visual_logits must be a floating [B,Q,C] tensor.')
        if visual_logits.shape[-1]!=self.refinement.num_classes:
            raise ValueError('visual_logits class count differs from the taxonomy.')
        if not isinstance(boxes,torch.Tensor) or boxes.shape!=(*visual_logits.shape[:2],4):
            raise ValueError('boxes must have shape [B,Q,4].')
        with torch.autocast(device_type=visual_logits.device.type,enabled=False):
            return self._forward_float32(visual_logits,boxes,modal_inputs)

    def _forward_float32(self, raw, boxes, modal_inputs):
        if modal_inputs is None or modal_inputs.get('weight_tokens_raw') is None:
            return self._empty(raw,boxes)
        # Explicitly whitelist telemetry; unrelated GT or application metadata
        # cannot become evidence or route inputs via a copied dictionary.
        modal={key:modal_inputs[key] for key in MODAL_KEYS if modal_inputs.get(key) is not None}
        tokens=modal['weight_tokens_raw'].to(device=raw.device,dtype=torch.float32)
        if tokens.ndim!=3 or tokens.shape[-1]!=7:
            raise ValueError('weight_tokens_raw must have shape [batch, events, 7].')
        mask=modal.get('weight_mask')
        if mask is None:mask=torch.ones(tokens.shape[:2],device=raw.device,dtype=torch.bool)
        else:
            # This native-forward check must precede bool conversion: NaN or
            # nonbinary masks must not silently turn into accepted events.
            if not bool((torch.isfinite(mask)&((mask==0)|(mask==1))).all()):
                raise ValueError('weight_mask must contain only boolean or 0/1 values.')
            mask=mask.to(device=raw.device,dtype=torch.bool)
        if mask.shape!=tokens.shape[:2]:raise ValueError('weight_mask must have shape [batch, events].')
        if tokens.shape[0]!=raw.shape[0]:raise ValueError('Modal batch differs from visual batch.')
        modal['weight_tokens_raw']=tokens;modal['weight_mask']=mask
        if tokens.shape[1]==0:
            # Native forward validates scalar/order shapes, then uses its empty
            # response; do not substitute a differently combined reason code.
            self.upstream._integrity_check(tokens,mask,modal)
            return self._empty(raw,boxes)
        logp,probability,scene,integrity=self.upstream.mass_evidence(tokens,mask,modal)
        residual=(logp-probability.clamp_min(1e-8).log()[:,None,:]).clamp(
            min=-self.upstream.evidence_clip,max=0.)
        z=raw.float();float_boxes=boxes.to(device=raw.device,dtype=torch.float32)
        b,q,c=raw.shape
        candidate_mask=torch.zeros(b,q,device=raw.device,dtype=torch.bool)
        lists=[]
        for batch in range(b):
            if not bool(integrity['gate'][batch]):
                lists.append(torch.empty(0,device=raw.device,dtype=torch.long));continue
            count=min(q,int(mask[batch].sum())+self.upstream.candidate_extra)
            chosen=self.upstream._select_candidates(z[batch],float_boxes[batch],count).to(raw.device)
            candidate_mask[batch,chosen]=True
            if self.matching_order=='query_order':chosen=chosen.sort().values
            lists.append(chosen)
        k=max((len(indices) for indices in lists),default=0)
        indices=torch.full((b,k),-1,device=raw.device,dtype=torch.long)
        slots=torch.zeros(b,k,device=raw.device,dtype=torch.bool)
        for batch,chosen in enumerate(lists):
            indices[batch,:len(chosen)]=chosen;slots[batch,:len(chosen)]=True
        # Exactly one call to the independent new association/readout module.
        refined=self.refinement(raw,indices,slots,scene,residual,mask,integrity['gate'])
        selected=refined['assigned_event']>=0
        native_margin=torch.full((b,q),-torch.inf,device=raw.device)
        for batch in range(b):
            if not bool(selected[batch].any()):continue
            rows=refined['row_margin'][batch,selected[batch]]
            cols=refined['col_margin'][batch,selected[batch]]
            if int(mask[batch].sum())<=1:rows=torch.full_like(rows,torch.inf)
            if len(lists[batch])<=1:cols=torch.full_like(cols,torch.inf)
            native_margin[batch,selected[batch]]=torch.minimum(rows,cols)
        scene_logits=raw.clone()
        mapping=self.refinement.leaf_to_parent.to(raw.device)
        alpha=self.refinement.readout_alpha.to(raw.device)
        for batch,chosen in enumerate(lists):
            if not bool(refined['applied_transaction'][batch]):continue
            current=z[batch,chosen];mean=current.clone()
            for parent in mapping.unique(sorted=True):
                leaves=(mapping==parent).nonzero().flatten()
                if len(leaves)>1:mean[:,leaves]=current[:,leaves].mean(-1,keepdim=True)
            scene_logits[batch,chosen]=(current-alpha*(current-mean)+2*scene[batch]).to(raw.dtype)
        output=dict(refined,pred_boxes=boxes,pred_logits_scene=scene_logits,
                    candidate_mask=candidate_mask,candidate_indices=indices,candidate_slot_mask=slots,
                    accepted_mask=selected,assignment_margin=native_margin,
                    scene_evidence=scene,scene_probability=probability,event_residual=residual,
                    event_log_probability=logp,integrity_gate=integrity['gate'],
                    integrity_rejection_code=integrity['rejection_code'],numeric_valid=integrity['numeric_valid'],
                    snapshot_independent=integrity['snapshot_independent'])
        for name in ('record_residual_g','baseline_residual_g','total_residual_g','end_residual_g',
                     'chain_residual_g','total_threshold_g','chain_threshold_g'):
            output[f'integrity_{name}']=integrity[name]
        return output


def bridge_from_existing_wrapper(wrapper, mapping, alpha=.5, *, matching_order='score_order'):
    """Return an optional isolated bridge; do not replace wrapper.cvsr/forward.

    A future hook belongs after decoder logits/boxes are produced and before the
    old self.cvsr call. Identity-first native wrappers are rejected to avoid
    consuming an already collapsed visual representation. A deepcopy prevents
    bridge device/mode changes from changing the supplied wrapper's CVSR object.
    """
    if getattr(wrapper,'identity_first',False):
        raise ValueError('Use raw decoder logits; an identity-first wrapper would collapse them twice.')
    upstream=getattr(wrapper,'cvsr',None)
    if upstream is None:raise ValueError('The existing wrapper must provide its original CVSR.')
    return StateNeutralCVSRBridge(copy.deepcopy(upstream),mapping,alpha,matching_order=matching_order)
