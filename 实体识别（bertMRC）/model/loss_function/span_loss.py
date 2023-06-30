import torch
from torch import nn
from utils.arguments_parse import args
class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,span_logits,span_label,seq_mask):
        span_label = span_label.view(size=(-1,))
        span_loss = self.loss_func(input=span_logits.view(size=(-1, 2)), target=span_label)

        avg_se_loss = torch.sum(span_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss

        