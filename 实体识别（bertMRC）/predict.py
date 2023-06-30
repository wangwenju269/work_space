import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
from utils.arguments_parse import args
import data_preprocessing
from model.model import bertMRC
from model.loss_function import multilabel_cross_entropy
from model.metrics import metrics
from data_preprocessing import *
import json
from tqdm import tqdm
import unicodedata, re
from data_preprocessing import predict_data_prepro
from data_preprocessing import tools
from tqdm import tqdm

device = torch.device('cuda')

added_token = ['[unused1]', '[unused2]']
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_path, additional_special_tokens=added_token)
predicate2id,id2predicate=tools.load_schema()
model = bertMRC(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
model.load_state_dict(torch.load(args.checkpoints))
model.eval()


def load_data(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        sentences=[]
        for line in lines:
            data=json.loads(line)
            sentences.append(data['text'])
        return sentences


def build_text(data):
    sentences=[]
    text=data['text']
    for p in data['spo_list']:
        for o in pso_dict[p['predicate']]:
            subject_type=o.split('|')[0]
            object_type=o.split('|')[1]
            tmp_sent=subject_type + '[unused1]' + p['predicate'] + '[unused2]' + object_type + '[SEP]无答案'+text
            sentences.append(tmp_sent)
    return sentences


def encoder(sent):

    encode_dict=tokenizer.encode_plus(sent,max_length=args.max_length,pad_to_max_length=True)
    encode_sent_list=encode_dict['input_ids']
    token_type_ids_list=encode_dict['token_type_ids']
    attention_mask_list=encode_dict['attention_mask']
    return encode_sent_list,token_type_ids_list,attention_mask_list


def get_actual_id(text,role_text_event_type):
    text_encode=tokenizer.encode(text)
    one_input_encode=tokenizer.encode(role_text_event_type)
    text_start_id=tools.search(text_encode[1:-1],one_input_encode) 
    text_end_id=text_start_id+len(text_encode)-1
    if text_end_id>args.max_length:
        text_end_id=args.max_length
    
    text_token=tokenizer.tokenize(text)
    text_mapping = tools.token_rematch().rematch(text,text_token)

    return text_start_id,text_end_id,text_mapping


def get_start_end_i(start_logits,end_logits,span_logits,text_start_id,text_end_id):

    arg_index = []
    i_start, i_end = [], []
    for i in range(text_start_id, text_end_id):
        if start_logits[i][1] > 0.48:
            i_start.append(i)
        if end_logits[i][1] > 0.48:
            i_end.append(i)
    # 然后遍历i_end
    cur_end = -1
    for e in i_end:
        s = []
        for i in i_start:
            if e >= i >= cur_end:
                s.append(i)
        max_s = 0.4
        t = None
        for i in s:
            if span_logits[i][e][1] > max_s:
                t = (i, e)
                max_s = span_logits[i][e][1]
        print('max_s:',max_s)
        cur_end = e
        if t is not None:
            arg_index.append(t)
    return arg_index


def extract_entity_from_start_end_ids(start_logits, end_logits,text_start_id, text_end_id):
        # 根据开始，结尾标识，找到对应的实体
    start_ids=[0 for i in range(len(start_logits))]
    end_ids=[0 for i in range(len(end_logits))]

    for i in range(text_start_id, text_end_id):
        if start_logits[i] > 0.48:
            start_ids[i] = 1
        if end_logits[i] > 0.48:
            end_ids[i] = 1

    start_end_tuple_list = []
    for i, start_id in enumerate(start_ids):
        if start_id == 0:
            continue
        if end_ids[i] == 1:
            start_end_tuple_list.append((i, i))
            continue
        j = i + 1
        find_end_tag = False
        while j < len(end_ids):
            # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
            if start_ids[j] == 1:
                break
            if end_ids[j] == 1:
                start_end_tuple_list.append((i, j))
                find_end_tag = True
                break
            else:
                j += 1
        if not find_end_tag:
            start_end_tuple_list.append((i, i))
    return start_end_tuple_list


def model_predict(sentence):

    input_ids, input_seg, input_mask = encoder(sentence)
    input_ids = torch.Tensor([input_ids]).long()
    input_seg = torch.Tensor([input_seg]).long()
    input_mask = torch.Tensor([input_mask]).float()
    sub_start_logit,sub_end_logit,ob_start_logit,ob_end_logit = model( 
                input_ids=input_ids.to(device), 
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                is_training=False
                )
    sub_start_logit=sub_start_logit.view(-1,).to(torch.device('cpu')).detach().numpy().tolist()
    sub_end_logit=sub_end_logit.view(-1,).to(torch.device('cpu')).detach().numpy().tolist()
    ob_start_logit=ob_start_logit.view(-1,).to(torch.device('cpu')).detach().numpy().tolist()
    ob_end_logit=ob_end_logit.view(-1,).to(torch.device('cpu')).detach().numpy().tolist()
    
    return sub_start_logit,sub_end_logit,ob_start_logit,ob_end_logit


def extract_arg(start_logits,end_logits,text_start_id,text_end_id,text_mapping,text):
    # args_index = get_start_end_i(start_logits[i],end_logits[i],span_logits[i],text_start_id,text_end_id)
    args_index = extract_entity_from_start_end_ids(start_logits,end_logits,text_start_id,text_end_id)
    # args_index=sapn_decode(span_logits[i])
    one_role_args=[]
    for k in args_index:
        if len(text_mapping)>3:
            dv = 0
            while  k[0]-text_start_id+dv<len(text_mapping) and k[0]-text_start_id+dv>=0 and text_mapping[k[0]-text_start_id+dv] == []:
                dv+=1

            start_split=text_mapping[k[0]-text_start_id+dv] if k[0]-text_start_id+dv<len(text_mapping) and k[0]-text_start_id+dv>=0 else []
            dv = 0
            while k[1]-text_start_id+dv<len(text_mapping) and k[1]-text_start_id+dv>=0 and text_mapping[k[1]-text_start_id+dv] == []:
                dv-=1
            end_split=text_mapping[k[1]-text_start_id+dv] if k[1]-text_start_id+dv<len(text_mapping) and k[1]-text_start_id+dv>=0 else []
            if start_split !=[] and end_split !=[]:
                tmp=text[start_split[0]:end_split[-1]+1]
                one_role_args.append(tmp)
    return one_role_args

multi_rel=['配音','上映时间','票房','获奖','饰演']

def main():
    sentences = load_data('./data/duie_test1.json')
    # sentences = load_data(args.dev_path)
    with open('./output/one_stage.json','w',encoding='utf-8') as f:
        count=0
        for sent in tqdm(sentences):
            count+=1
            if count>-1:
                sub_start_logits,sub_end_logits,ob_start_logits,ob_end_logits=model_predict(sent)
                spo_args=[]

                text_start_id,text_end_id,text_mapping=get_actual_id(sent,sent)
                sub_one_args=extract_arg(sub_start_logits,sub_end_logits,text_start_id,text_end_id,text_mapping,sent)
                ob_one_args=extract_arg(ob_start_logits,ob_end_logits,text_start_id,text_end_id,text_mapping,sent)

                tmp_result=dict()
                tmp_result['text']=sent
                tmp_result['s_list']=sub_one_args
                tmp_result['o_list']=ob_one_args
                json_data=json.dumps(tmp_result,ensure_ascii=False)
                f.write(json_data+'\n')
            
if __name__ == '__main__': 
    main()