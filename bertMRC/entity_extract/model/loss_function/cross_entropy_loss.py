import torch
from torch import nn
from utils.arguments_parse import args

class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,start_logits,end_logits,start_label, end_label,seq_mask):
        start_label, end_label = start_label.view(size=(-1,)), end_label.view(size=(-1,))
        start_loss = self.loss_func(input=start_logits.view(size=(-1, 2)), target=start_label)
        end_loss = self.loss_func(input=end_logits.view(size=(-1, 2)), target=end_label)

                    
        sum_loss = start_loss + end_loss
        sum_loss *= seq_mask.view(size=(-1,))

        avg_se_loss = torch.sum(sum_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss

class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,logits,label):
        label = label.view(size=(-1,))
        loss = self.loss_func(input=logits.view(size=(-1, 56)), target=label)
        return loss