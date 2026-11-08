"""Isolated RGB inference with canonical CPU-FP32 state-contrast refinement."""
from __future__ import annotations

import copy
import torch
from torch import nn

if __package__:
    from .native_bridge import bridge_from_existing_wrapper, MODAL_KEYS
else:
    from native_bridge import bridge_from_existing_wrapper, MODAL_KEYS


class StateNeutralDetector(nn.Module):
    """Frozen GPU visual detector -> CPU fusion once -> original output device.

    The supplied native model is copied unless ``copy_detector=False``. The
    original CVSR forward is disconnected before image inference. Physical
    evidence, candidate selection and new refinement are CPU-FP32 operations,
    matching the locked research-output protocol. Transfers are explicit and
    must be included when timing this wrapper. This is inference-only.
    """

    def __init__(self, detector, mapping, alpha=.5, *, copy_detector=True):
        super().__init__()
        self.bridge = bridge_from_existing_wrapper(detector, mapping, alpha).cpu().eval()
        self.detector = copy.deepcopy(detector) if copy_detector else detector
        self.detector.cvsr = None
        self.requires_grad_(False)
        self.eval()

    def visual(self, x):
        return self.detector(x, targets=None, modal_inputs=None)

    @torch.inference_mode()
    def refine(self, visual, modal_inputs=None):
        logits = visual['pred_logits_cfqd']
        device = logits.device
        modal = None if modal_inputs is None else {
            key: modal_inputs[key].detach().cpu()
            for key in MODAL_KEYS if modal_inputs.get(key) is not None
        }
        fusion = self.bridge(logits.detach().cpu(), visual['pred_boxes'].detach().cpu(), modal)
        result = dict(visual)
        result['pred_logits'] = fusion['pred_logits'].to(device=device)
        result['pred_logits_cvsr'] = result['pred_logits']
        result['pred_logits_scene'] = fusion['pred_logits_scene'].to(device=device)
        # Keep diagnostics on their computation device. Detection outputs alone
        # are transferred back for the shared GPU postprocessor.
        result.update({f'state_{key}': value for key, value in fusion.items()
                       if key not in ('pred_logits', 'pred_logits_scene', 'pred_boxes')})
        return result

    @torch.inference_mode()
    def forward(self, x, targets=None, modal_inputs=None):
        if self.training:
            raise ValueError('StateNeutralDetector is inference-only.')
        if targets is not None:
            raise ValueError('Ground truth is not an inference input.')
        return self.refine(self.visual(x), modal_inputs)
