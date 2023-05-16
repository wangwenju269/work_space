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

class bertMRC(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.roberta_encoder.resize_token_embeddings(len(tokenizer))

        self.s_startlayer = torch.nn.Linear(in_features=768, out_features=1)
        self.s_endlayer = torch.nn.Linear(in_features=768, out_features=1)
        self.o_startlayer = torch.nn.Linear(in_features=768, out_features=1)
        self.o_endlayer = torch.nn.Linear(in_features=768, out_features=1)

        self.sigmoid=nn.Sigmoid()

    def forward(self, input_ids, input_mask, input_seg, is_training=False):
        bert_output = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)  # (bsz, seq, dim)
        encoder_rep = bert_output[0]

        s_startlogits=self.sigmoid(self.s_startlayer(encoder_rep))
        s_endlogits=self.sigmoid(self.s_endlayer(encoder_rep))

        o_startlogits=self.sigmoid(self.o_startlayer(encoder_rep))
        o_endlogits=self.sigmoid(self.o_endlayer(encoder_rep))

        if is_training:
            return s_startlogits,s_endlogits,o_startlogits,o_endlogits
        else:
            return s_startlogits,s_endlogits,o_startlogits,o_endlogits

