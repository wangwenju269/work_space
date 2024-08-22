import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, labels):
        y_pred = torch.argmax(logits,dim=-1)
        y_pred = y_pred.view(size=(-1,)).float()
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred,y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc
