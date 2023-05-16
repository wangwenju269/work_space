import torch
from torch import nn
device = torch.device('cuda')


class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        logits=torch.argmax(logits,dim=-1)
        ones=torch.ones_like(labels)
        zero=torch.zeros_like(labels)
        y_ture_mask=torch.where(labels<0.5,zero,ones)
        y_pred_mask=torch.where(logits<0.5,zero,ones)

        y_ture_mask=y_ture_mask.view(size=(-1,))
        y_pred_mask=y_pred_mask.view(size=(-1,))

        y_pred=logits.view(size=(-1,)).float()
        y_true=labels.view(size=(-1,)).float()
        corr=torch.eq(y_pred,y_true)
        acc=torch.sum(corr.float())/labels.size()[0]
        corr=torch.multiply(corr.float(),y_ture_mask)
        recall=torch.sum(corr)/(torch.sum(y_ture_mask)+1e-8)
        precision=torch.sum(corr)/(torch.sum(y_pred_mask)+1e-8)
        f1=2*recall*precision/(recall+precision+1e-8)

        return acc, precision, f1
