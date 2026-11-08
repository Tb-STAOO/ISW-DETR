"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', 'RTDETRP2Guide', 'ScaleVisionMMRTDETR']


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


@register()
class ScaleVisionMMRTDETR(RTDETR):
    """RT-DETR wrapper that forwards external multimodal inputs to the decoder."""

    def forward(self, x, targets=None, modal_inputs=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets=targets, modal_inputs=modal_inputs)
        return x


@register()
class RTDETRP2Guide(RTDETR):
    """RT-DETR wrapper for encoders that return a training-only P2 guide loss."""

    def forward(self, x, targets=None):
        x = self.backbone(x)
        encoded = self.encoder(x)
        if self.training:
            x, p2_guide_loss = encoded
        else:
            x = encoded
            p2_guide_loss = None

        outputs = self.decoder(x, targets)
        if self.training and p2_guide_loss is not None:
            outputs['training_aux_losses'] = {
                'loss_p2_guide': p2_guide_loss,
            }
        return outputs
