import os
import sys
sys.path.append('./')
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
from tqdm import tqdm
from sklearn.utils import shuffle

tokenizer=tools.get_tokenizer()
predicate2id,id2predicate,s_entity_type,o_entity_type,_,_=tools.load_schema()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        result=[]
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            text = text
            s_dict={}
            o_dict={}
            spo_dict={}
            for spo in data['spo_list']:
                s=spo['subject']
                s_dict[spo['subject_type']]=spo['subject']
                p=spo['predicate']
                tmp_ob_type=[v for k,v in spo['object_type'].items()]
                tmp_ob=[v for k,v in spo['object'].items()]
                for i in range(len(tmp_ob)):
                    p_o=p+'|'+tmp_ob_type[i]
                    spo_dict[s+'|'+tmp_ob[i]]=p_o
                    o_dict[tmp_ob_type[i]]=tmp_ob[i]
            for sk,sv in s_dict.items():
                for ok,ov in o_dict.items():
                    s_flag=s_entity_type[sk]
                    o_flag=o_entity_type[ok]
                    s_start=tools.search(sv,text)
                    s_end=s_start+len(sv)
                    text2=text[:s_start]+s_flag[0]+sv+s_flag[1]+text[s_end:]
                    o_start=tools.search(ov,text2)
                    o_end=o_start+len(ov)
                    text3=text2[:o_start]+o_flag[0]+ov+o_flag[1]+text2[o_end:]
                    if sv+'|'+ov in spo_dict.keys():
                        labels=predicate2id[spo_dict[sv+'|'+ov]]
                    else:
                        labels=0
                    result.append({'text':text3,'labels':labels,'flag':(s_flag[0],o_flag[0])})
        return result

def encoder(sentence, flag):
    encode_dict = tokenizer.encode_plus(sentence,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']

    s_encode = tokenizer.encode(flag[0])
    s_start_idx = tools.search(s_encode[1:-1], encode_sent)
    
    o_encode = tokenizer.encode(flag[1])
    o_start_idx = tools.search(o_encode[1:-1], encode_sent)
    flag=[s_start_idx,o_start_idx]

    # flag = [i for i,v in enumerate(encode_sent) if v==2]
    # while len(flag)<2:
    #     flag.append(0)

    return encode_sent, token_type_ids, attention_mask, flag


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_ids, input_seg, input_mask, flag = encoder(
            item['text'], item['flag'])

        one_data = {
            "input_ids": torch.tensor(input_ids).long(),
            "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(),
            "labels": torch.tensor(item['labels']).long(),
            "flag": torch.tensor(flag).long()
        }
        return one_data


def yield_data(file_path):
    tmp = MyDataset(load_data(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':

    data = load_data(args.train_path)
    print(data[0])

    # print(input_ids_list[0])
    # print(token_type_ids_list[0])
    # print(start_labels[0])
