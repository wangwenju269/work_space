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

tokenizer=tools.get_tokenizer()

model = bertMRC(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
model.load_state_dict(torch.load(args.checkpoints))
model.eval()
predicate2id,id2predicate,s_entity_type,o_entity_type,otype,ptype=tools.load_schema()

def load_data(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        sentences=[]
        for line in lines:
            data=json.loads(line)
            sentences.append(data)
        return sentences


def build_text(data):
    sentences=[]
    text=data['text']
    spo_list=[]
    flag=[]
    # print(data['s_list'])
    for s in data['s_list']:
        for o in data['o_list']:
            if s==o:
                continue
            s_flag=['[unused4]','[unused5]']
            o_flag=['[unused56]','[unused57]']
            s_start=tools.search(s,text)
            s_end=s_start+len(s)
            text2=text[:s_start]+s_flag[0]+s+s_flag[1]+text[s_end:]
            o_start=tools.search(o,text2)
            o_end=o_start+len(o)
            text3=text2[:o_start]+o_flag[0]+o+o_flag[1]+text2[o_end:]
            sentences.append(text3)
            spo_list.append({'subject':s,'object':o})
            flag.append((s_flag[0],o_flag[0]))

    return sentences,spo_list,flag


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
    flag2=[s_start_idx,o_start_idx]


    return encode_sent, token_type_ids, attention_mask, flag2


def model_predict(input_text_list,flag_list):
    logits=[]
    input_ids2 = []
    input_seg2 = []
    input_mask2 = []
    flag2 = []
    for i in range(len(input_text_list)):
        sentence = input_text_list[i]
        input_ids1, input_seg1, input_mask1,flag1 = encoder(sentence,flag_list[i])
        input_ids2.append(input_ids1)
        input_seg2.append(input_seg1)
        input_mask2.append(input_mask1)
        flag2.append(flag1)

    batch_size=20
    start=0
    end=batch_size
    while start<len(input_ids2):
        input_ids = torch.Tensor(input_ids2[start:end]).long()
        input_seg = torch.Tensor(input_seg2[start:end]).long()
        input_mask = torch.Tensor(input_mask2[start:end]).float()
        flag = torch.Tensor(flag2[start:end]).long()
        start+=batch_size
        end+=batch_size
        logit = model( 
                    input_ids=input_ids.to(device), 
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device),
                    flag=flag.to(device),
                    is_training=False
                    )
        logit=torch.argmax(logit,dim=-1).view(-1,)
        logit=logit.to(torch.device('cpu')).detach().numpy().tolist()
        logits.extend(logit)

    return logits




multi_rel=['配音','上映时间','票房','获奖','饰演']

def main():
    sentences= load_data('./output/one_stage.json')

    with open('./output/duie.json','w',encoding='utf-8') as f:
        count=0
        for sent in tqdm(sentences):
            count+=1
            if count>-1:
                input_text_list,spo_list,flag_list=build_text(sent)

                logits=model_predict(input_text_list,flag_list) if input_text_list!=[] else []
                for i in range(len(spo_list)):
                    if logits[i]>0:
                        spo_list[i]['predicate']=id2predicate[logits[i]]

                multi_predicate=[]
                for s in spo_list:    #先找出所有存在多个object类型的关系predicate
                    if 'predicate' in s.keys():
                        if s['predicate'].split('|')[0] in multi_rel:
                            multi_predicate.append(s['predicate'].split('|')[0])
                
                new_spo_list=[]
                for s in spo_list:
                    if 'predicate' in s.keys():
                        if s['predicate'].split('|')[0] not in multi_rel: #先保存所有不存在多个object类型的关系predicate
                            tmp={}
                            tmp['subject_type']=ptype[s['predicate']]['subject_type']
                            tmp['object_type']={'@value':ptype[s['predicate']]['object_type']}
                            tmp['predicate']=s['predicate'].split('|')[0]
                            tmp['subject']=s['subject']
                            tmp['object']={'@value':s['object']}
                            new_spo_list.append(tmp)

                for p in multi_predicate: #合并一个predicate存在的多个object类型的关系
                    tmp_p=[]
                    for s in spo_list:
                        if 'predicate' in s.keys():
                            if s['predicate'].split('|')[0] == p:
                                tmp_p.append(s)
                    if tmp_p !=[]:
                        new_spo={}
                        new_spo['subject_type']=ptype[tmp_p[0]['predicate']]['subject_type']
                        new_spo['subject']=tmp_p[0]['subject']
                        new_spo['predicate']=tmp_p[0]['predicate'].split('|')[0]
                        ob_type={}
                        ob={}
                        for q in tmp_p:
                            if tmp_p[0]['predicate'].split('|')[0]+'|'+ptype[q['predicate']]['object_type'] in otype.keys():
                                ob_type[otype[tmp_p[0]['predicate'].split('|')[0]+'|'+ptype[q['predicate']]['object_type']]]=ptype[q['predicate']]['object_type']
                                ob[otype[tmp_p[0]['predicate'].split('|')[0]+'|'+ptype[q['predicate']]['object_type']]]=q['object']
                            else:
                                ob_type['@value']=ptype[q['predicate']]['object_type']
                                ob['@value']=q['object']
                            
                        new_spo['object_type']=ob_type
                        new_spo['object']=ob
                    new_spo_list.append(new_spo)
                
                new_spo_list2=[]
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)
                
                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]
                    if '@value' not in new_spo_list2[i]['object'].keys():
                        new_spo_list2[i]['object']['@value']='flag'
                    if '@value' not in new_spo_list2[i]['object_type'].keys():
                        new_spo_list2[i]['object_type']['@value']='人物'

                tmp_result=dict()
                tmp_result['text']=sent['text']
                tmp_result['spo_list']=new_spo_list2
                json_data=json.dumps(tmp_result,ensure_ascii=False)
                f.write(json_data+'\n')
            
if __name__ == '__main__': 
    main()