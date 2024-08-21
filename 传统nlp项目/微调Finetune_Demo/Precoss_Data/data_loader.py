from torch.utils.data import Dataset,DataLoader
import json
from transformers import AutoTokenizer
import torch
from Utils.argpaser import parse_args
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        with open(data_file, 'r',encoding='utf-8') as f:
            all_data = {}
            tablename = []
            for idx, line in enumerate(f):
                if idx == 0:
                   tablename = line.strip('\n').split(',')
                   print(tablename)
                else:
                   data = line.strip('\n').split(',')
                   all_data[idx-1] = {tablename[i]:data[i].lower() for i in range(3)}
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_1 = self.data[idx]['CDR3']
        sentence_2 = self.data[idx]['Epitope']
        features = tokenizer( sentence_1,
                              sentence_2,
                              padding = 'max_length',
                              max_length = args.max_seq_length,
                              truncation = True,
                              return_tensors='pt')
        label = self.data[idx]['Class_label']
        label = torch.tensor(int(label))
        return features,label


def yield_data(file_path):
    data = AFQMC(file_path)
    return DataLoader(data, batch_size=args.batch_size, shuffle=True)

if __name__ == '__main__':
    train_data = AFQMC(args.train_file)
    train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True)
    for x in train_dataloader:
        print(x)
        break

