import os
import sys
# sys.path.append('./')
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re
from data_preprocessing import tools

tokenizer=tools.get_tokenizer()


def load_data(file_path):
    event_type_dict=tools.load_schema()
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        sentences=[]
        for line in lines:
            data=json.loads(line)
            text=data['text']
            title=data['title']
            if 'event_list' in data.keys() and data['event_list'] !=[]:
                for event in data['event_list']:
                    event_type = event['event_type']
                    if event_type !='无事件':
                        role_list = event_type_dict[event_type]
                        for role in role_list:
                            sent = event_type+'[unused1]'+role+'[SEP]'+text
                            sentences.append(sent)
        return sentences

def encoder(sentences):
    encode_sent_list=[]
    token_type_ids_list=[]
    attention_mask_list=[]
    for sent in sentences:
        encode_dict=tokenizer.encode_plus(sent,max_length=args.max_length,pad_to_max_length=True)
        encode_sent_list.append(encode_dict['input_ids'])
        token_type_ids_list.append(encode_dict['token_type_ids'])
        attention_mask_list.append(encode_dict['attention_mask'])
    return encode_sent_list,token_type_ids_list,attention_mask_list
