from data_pre_process.tools import tokenizer
import tqdm
import sys
sys.path.append('./')
from utils.arguments_parse import args
import json
import torch
from torch.utils.data import DataLoader, Dataset

classify = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
label2id = { x:i for i, x in enumerate(classify)}

class myDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_length
        self.data = self._read_data()

    def _read_data(self):

        with open(self.data_file, 'r', encoding='UTF-8') as f:
            text = []
            for l in f:
                l = json.loads(l)
                classify_name = l['event_list'][0]['class']
                text.append({'text': l['text'],'label':label2id[classify_name]})
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]['text']
        label =self.data[index]['label']
        input_ids, token_type_ids, attention_mask = self.encoder(item)
        one_data = {
            "input_ids": torch.tensor(input_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).long(),
            "label": torch.tensor(label).long()
        }
        return one_data

    def encoder(self,sentence):
        encode_dict = tokenizer.encode_plus(sentence, truncation=True,
                                                       max_length=args.max_length,
                                                       padding='max_length')
        input_ids = encode_dict['input_ids']
        token_type_ids = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']
        return input_ids, token_type_ids, attention_mask


class My_Dataset():
    def __init__(self, data_file, tokenizer, max_length):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_length
        self.data = self._read_data()

    def _read_data(self):
        with open(self.data_file, 'r', encoding='UTF-8') as f:
            text = []
            for l in f:
                l = json.loads(l)
                classify_name = l['event_list'][0]['class']
                text.append({'text': l['text'],'label':label2id[classify_name]})
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = self.data[index]['text']
        label =self.data[index]['label']
        features = self.tokenizer(sent, truncation=True,
                                  max_length=args.max_length,
                                  padding='max_length')

        input_ids = features["input_ids"]
        token_type_ids = features["token_type_ids"]
        attention_mask = features["attention_mask"]
        one_data = {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label": label}
        return one_data


def collate_fn(batch_data):
    input_ids =  [item["input_ids"] for item in batch_data]
    token_type_ids = [item["token_type_ids"] for item in batch_data]
    attention_mask = [item["attention_mask"] for item in batch_data]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    label = [item["label"] for item in batch_data]
    label = torch.tensor(label, dtype=torch.long)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,"label": label}


def yield_data(file_path):
    temp_data = My_Dataset(file_path,tokenizer,args.max_length)
    return DataLoader(temp_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

def yield_data1(file_path):
    temp_data = myDataset(file_path,tokenizer,args.max_length)
    return DataLoader(temp_data, batch_size=args.batch_size, shuffle=True)

if __name__ == '__main__':
    train_dataset = yield_data(args.train_path)
    for step, batch_data in enumerate(train_dataset):
        input_ids = batch_data["input_ids"]
        ken_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        label = batch_data['label']
        print(input_ids)
        break



