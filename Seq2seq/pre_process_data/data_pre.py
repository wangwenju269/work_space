import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from utils import arg
args = arg.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= 10:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = sample['chinese']
        labels = sample['english']
        data = self.encoding(inputs,labels)
        return data

    def encoding(self,inputs,labels):
        features =  tokenizer(inputs,
                            padding = "max_length",
                            max_length=args.max_input_length,
                            truncation=True,
                            return_tensors="pt")
        with tokenizer.as_target_tokenizer():
             labels =   tokenizer(labels,
                             padding = "max_length",
                             max_length=args.max_input_length,
                             truncation=True,
                             return_tensors="pt")
             labels = labels["input_ids"].squeeze()
        end_token_index = torch.where(labels == 0)[0]
        end_token_index = end_token_index.cpu().numpy().tolist()[0]
        labels[end_token_index:] = -100
        features['labels'] = labels
        return features

def Yield_data(file_path):
    data = TRANS(file_path)
    return DataLoader(data, batch_size=args.batch_size, shuffle=True)


'''
第2种写法, 利用collate_fn
'''
class trans:
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= 10:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs,
        padding='max_length',
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding='max_length',
            max_length=args.max_target_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1 :] = -100
        batch_data['labels'] = labels
    return batch_data

def yield_data(file_path,shuffle=True):
    data = trans(file_path)
    return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,collate_fn= collote_fn)


if __name__=='__main__':
   data = yield_data(args.train_file)
   batch = next(iter(data))
   print(batch.keys())
   print('batch shape:', {k: v.shape for k, v in batch.items()})
   print(batch)


# def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
#     def collote_fn(batch_samples):
#         batch_inputs, batch_targets = [], []
#         for sample in batch_samples:
#             batch_inputs.append(sample['chinese'])
#             batch_targets.append(sample['english'])
#         batch_data = tokenizer(
#             batch_inputs,
#             padding=True,
#             max_length=args.max_input_length,
#             truncation=True,
#             return_tensors="pt"
#         )
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(
#                 batch_targets,
#                 padding=True,
#                 max_length=args.max_target_length,
#                 truncation=True,
#                 return_tensors="pt"
#             )["input_ids"]
#             batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
#             end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
#             for idx, end_idx in enumerate(end_token_index):
#                 labels[idx][end_idx + 1:] = -100
#             batch_data['labels'] = labels
#         return batch_data
#
#     return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle,
#                       collate_fn=collote_fn)
