import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel,LongformerModel
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
from data_preprocessing import tools

tokenizer=tools.get_tokenizer()
device = torch.device('cuda')


class bertMRC(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.roberta_encoder.resize_token_embeddings(len(tokenizer))
        self.cls_layer = torch.nn.Linear(in_features=768*2, out_features=56)

    def forward(self, input_ids, input_mask, input_seg,flag, is_training=False):
        bert_output = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)  # (bsz, seq, dim)
        encoder_rep = bert_output[0]
        batch_size,_,hidden = encoder_rep.shape
        entity_encode=torch.Tensor(batch_size,2*hidden)
        for i in range(batch_size):
            start_entity=encoder_rep[i,flag[i,0],:].view(hidden,)
            end_entity=encoder_rep[i,flag[i,1],:].view(hidden,)
            span_matrix = torch.cat([start_entity, end_entity], dim=-1)
            entity_encode[i]=span_matrix
        entity_encode=entity_encode.to(device)
        logits = self.cls_layer(entity_encode)

        if is_training:
            return logits
        else:
            return torch.nn.functional.softmax(logits, dim=-1)

