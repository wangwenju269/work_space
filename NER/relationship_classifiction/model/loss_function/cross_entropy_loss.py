import torch
from torch import nn
from utils.arguments_parse import args



class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,logits,label):
        label = label.view(size=(-1,))
        loss = self.loss_func(input=logits.view(size=(-1, 56)), target=label)
        return loss