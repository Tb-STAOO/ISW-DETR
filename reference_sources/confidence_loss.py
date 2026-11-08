import torch
from torch.nn import functional as F
import train_benefit_router as tr

def confidence_loss(model,batch,clean_batch):
    details=tr.forward(model,batch);pred=details['pred_logits']
    labels=batch['labels'];safe=labels.clamp_min(0)
    fg=batch['candidate_mask']&(labels>=0)
    bg=batch['candidate_mask']&batch['true_background_mask'];supervised=fg|bg
    def ce(z):return F.cross_entropy(z.transpose(1,2),safe,reduction='none')
    def target(z):
        foreground=z.softmax(-1).gather(-1,safe[...,None]).squeeze(-1)
        background=torch.exp(-F.softplus(z).mean(-1))
        return torch.where(bg,background,foreground).detach()
    truth=torch.stack([target(batch['z']),target(batch['base'])],-1)
    squared=(details['confidence']-truth).square().sum(-1)
    confidence=tr.row_mean(squared,supervised).mean()
    classification=tr.row_mean(ce(pred),fg).mean()
    background=tr.row_mean(F.softplus(pred).mean(-1),bg).mean()
    clean=tr.forward(model,clean_batch)['pred_logits']
    clean_loss=tr.row_mean((clean-clean_batch['base']).square().mean(-1),clean_batch['candidate_mask']).mean()
    loss=classification+.5*confidence+.1*background+.02*clean_loss
    return loss,dict(loss=float(loss.detach()),class_ce=float(classification.detach()),
        confidence_mse=float(confidence.detach()),background_bce=float(background.detach()),
        clean_regularizer=float(clean_loss.detach()),foreground=int(fg.sum()),background=int(bg.sum()),
        gate_mean=float(details['gate'][batch['candidate_mask']].mean().detach()))

