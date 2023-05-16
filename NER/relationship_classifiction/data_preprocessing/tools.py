import os
import sys
sys.path.append('./')
from transformers import BertTokenizer,LongformerTokenizer,AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re

def get_tokenizer():
    """添加特殊中文字符和未使用的token【unused1】"""
    added_token=['[unused'+str(i+1)+']' for i in range(99)]
    # path=args.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path,additional_special_tokens=added_token)
    special_tokens_dict = {'additional_special_tokens':['”','“']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

tokenizer=get_tokenizer()

class token_rematch:
    def __init__(self):
        self._do_lower_case = True


    @staticmethod
    def stem(token):
            """获取token的“词干”（如果是##开头，则自动去掉##）
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """控制类字符判断
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """判断是不是有特殊含义的符号
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return 0


def load_schema(file_path=args.schema_path):
    with open(file_path,'r',encoding='utf8') as f:
        lines=f.readlines() 
        schema_list=[]
        predicate_list=[]
        s_entity=[]
        o_entity=[]
        otype={}
        ptype={}
        for line in lines:
            data=json.loads(line)
            if data['subject_type'] not in s_entity:
                    s_entity.append(data['subject_type'])
            for k,v in data['object_type'].items():
                predicate_list.append(data['predicate']+'|'+v)
                otype[data['predicate']+'|'+v]=k
                ptype[data['predicate']+'|'+v]={'subject_type':data['subject_type'],'object_type':v}
                if v not in o_entity:
                    o_entity.append(v)

        s_entity_type={}
        s=1
        for i,e in enumerate(s_entity):
            s_entity_type[e]=(f'[unused{str(2*(s+1))}]',f'[unused{str(2*(s+1)+1)}]')
        

        o_entity_type={}
        o=10
        for i,e in enumerate(o_entity):
            o_entity_type[e]=(f'[unused{str(2*(o+len(s_entity_type)+1))}]',f'[unused{str(2*(o+len(s_entity_type)+1)+1)}]')


        predicate2id={}
        id2predicate={}

        for i,v in enumerate(predicate_list):
            predicate2id[v]=i+1
            id2predicate[i+1]=v

        return predicate2id,id2predicate,s_entity_type,o_entity_type,otype,ptype




if __name__=='__main__':
    schema=load_schema()
    print(len(schema))
    # cl=token_rematch()
    # d=cl.rematch(s,c)
    # print(d)