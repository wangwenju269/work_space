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
predicate2id,id2predicate=tools.load_schema()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        spo_lists = []
        result=[]
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            one_spo_list=[]
            for spo in data['spo_list']:
                s=spo['subject']
                p=spo['predicate']
                tmp_ob_type=[v for k,v in spo['object_type'].items()]
                tmp_ob=[v for k,v in spo['object'].items()]
                for i in range(len(tmp_ob)):
                    p_o=p+'|'+tmp_ob_type[i]
                    one_spo_list.append((s,p_o,tmp_ob[i]))

            sentences.append(text)
            spo_lists.append(one_spo_list)
            result.append({'text':text,'spo_list':one_spo_list})
        return result


def encoder(sentence, spo_list):
    encode_dict = tokenizer.encode_plus(sentence,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']

    s_startlabel = np.zeros([args.max_length]).astype(int) 
    s_endlabel = np.zeros([args.max_length]).astype(int)  

    o_startlabel = np.zeros([args.max_length]).astype(int) 
    o_endlabel = np.zeros([args.max_length]).astype(int) 

    for spo in spo_list:
        s_encode = tokenizer.encode(spo[0])
        s_start_idx = tools.search(s_encode[1:-1], encode_sent)
        s_end_idx = s_start_idx + len(s_encode[1:-1]) - 1
        s_startlabel[s_start_idx]=1
        s_endlabel[s_end_idx]=1

        o_encode = tokenizer.encode(spo[2])
        o_start_idx = tools.search(o_encode[1:-1], encode_sent)
        o_end_idx = o_start_idx + len(o_encode[1:-1]) - 1
        o_startlabel[o_start_idx]=1
        o_endlabel[o_end_idx]=1

    return encode_sent, token_type_ids, attention_mask, s_startlabel,s_endlabel,o_startlabel,o_endlabel


def data_pre(file_path):
    sentences, spo_lists = load_data(file_path)
    data = []
    for i in tqdm(range(len(sentences))):
        encode_sent, token_type_ids, attention_mask, s_startlabel,s_endlabel,o_startlabel,o_endlabel= encoder(
            sentences[i], spo_lists[i])
        tmp = {}
        tmp['input_ids'] = encode_sent
        tmp['input_seg'] = token_type_ids
        tmp['input_mask'] = attention_mask
        tmp['s_startlabel'] = s_startlabel
        tmp['s_endlabel'] = s_endlabel
        tmp['o_startlabel'] = o_startlabel
        tmp['o_endlabel'] = o_endlabel
        data.append(tmp)
    return data

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_ids, input_seg, input_mask, s_startlabel,s_endlabel,o_startlabel,o_endlabel = encoder(
            item['text'], item['spo_list'])

        one_data = {
            "input_ids": torch.tensor(input_ids).long(),
            "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(),
            "s_startlabel": torch.tensor(s_startlabel).long(),
            "s_endlabel": torch.tensor(s_endlabel).long(),
            "o_startlabel": torch.tensor(o_startlabel).long(),
            "o_endlabel": torch.tensor(o_endlabel).long()
        }
        return one_data


def yield_data(file_path):
    tmp = MyDataset(load_data(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':

    data = data_pre(args.train_path)
    print(data[0])

    # print(input_ids_list[0])
    # print(token_type_ids_list[0])
    # print(start_labels[0])
