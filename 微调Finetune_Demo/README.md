# <center>微调预训练模型

> *以同义句判断任务为例（每次输入两个句子，判断它们是否为同义句），构建一个 Transformers 模型DEMO案例。选择蚂蚁金融语义相似度数据集 [AFQMC](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip) ，标签 0 表示非同义句，1 表示同义句。*

- ### **Data Loader**

使用`Dataset` 类和 `DataLoader` 类处理数据集和加载样本，通过 `DataLoader` 按批加载数据，将样本转换成模型可以接受的输入格式。也就是将每个 batch 中的文本按照预训练模型的格式进行编码，批处理函数 `collate_fn` 来实现

```python
from torch.utils.data import Dataset,DataLoader
import json
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import torch
class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                data[idx] = sample
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

'''如果数据集非常巨大，难以一次性加载到内存中，可以继承 IterableDataset 类构建迭代型数据集.'''
class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file
    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

'''
DataLoader: DataLoader 库按批 (batch) 加载数据，并且将样本转换成模型可以接受的输入格式
'''
checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def collate_fn(batch_samples):
    batch_sentence_1,batch_sentence_2 = [],[]
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(sample['label'])
    X = tokenizer(batch_sentence_1,
                  batch_sentence_2,
                  padding=True,
                  truncation=True,
                  return_tensors='pt')
    y = torch.tensor(batch_label)
    return X,y
if __name__ == '__main__':
    train_data = AFQMC('data/train.json')
    print(train_data[0])
    train_data = IterableAFQMC('data/train.json')
    print(next(iter(train_data)))
    train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True,collate_fn = collate_fn)
    
```

- ### **Model**

  两种继承实现方式，如果通过torch的`nn.Module`继承，则后续无法获取`Transformers`库中`Config`配置参数及函数方法等，普遍采用继承`BertPreTrainedModel`,只需要通过预置的 `from_pretrained` 函数来加载模型参数。

  ```python
  import torch
  from torch import nn
  from transformers import AutoModel
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  checkpoint = 'bert-base-chinese'
  class  BertForPairwiseCLS(nn.Module):
         def __init__(self):
             super(BertForPairwiseCLS,self).__init__()
             self.bert_encoder = AutoModel.from_pretrained(checkpoint)
             self.dropout = nn.Dropout(0.1)
             self.classifier = nn.Linear(768,2)
         def forward(self,x):
             bert_out = self.bert_encoder(**x)
             cls_vectors = bert_out.last_hidden_state[:,0,:]
             cls_vectors = self.dropout(cls_vectors)
             logits = self.classifier(cls_vectors)
             return logits
  '上述写法无法再调用 Transformers 库预置的模型函数'
  from transformers import AutoConfig
  from  transformers import BertPreTrainedModel,BertModel
  config = AutoConfig.from_pretrained(checkpoint)
  class  BertForPairwiseCLS1(BertPreTrainedModel):
         def __init__(self,config):
             super().__init__(config)
             self.bert = BertModel(config, add_pooling_layer=False)
             self.dropout = nn.Dropout(config.hidden_dropout_prob)
             self.classifier = nn.Linear(768, 2)
             self.post_init()
         def forward(self, x):
             bert_output = self.bert(**x)
             cls_vectors = bert_output.last_hidden_state[:, 0, :]
             cls_vectors = self.dropout(cls_vectors)
             logits = self.classifier(cls_vectors)
             return logits
  
  ```

  ### **Train**

  训练模型时，将每一轮 Epoch 分为训练循环和验证循环，在训练循环中计算损失、优化模型的参数，在验证循环中评估模型的性能。

  ```python
  from tqdm.auto import tqdm
  def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
      progress_bar = tqdm(range(len(dataloader)))
      progress_bar.set_description(f'loss: {0:>7f}')
      finish_step_num = (epoch-1)*len(dataloader)
      model.train()
      for step, (X, y) in enumerate(dataloader, start=1):
          X, y = X.to(device), y.to(device)
          pred = model(X)
          loss = loss_fn(pred, y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          lr_scheduler.step()
          total_loss += loss.item()
          progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
          progress_bar.update(1)
      return total_loss
  def test_loop(dataloader, model, mode='Test'):
      assert mode in ['Valid', 'Test']
      size = len(dataloader.dataset)
      correct = 0
      model.eval()
      with torch.no_grad():
          for X, y in dataloader:
              X, y = X.to(device), y.to(device)
              pred = model(X)
              correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      correct /= size
      print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
  ```

  优化器：`AdamW`

  ```
  from transformers import AdamW
  optimizer = AdamW(model.parameters(), lr=5e-5)
  ```

  调度器：`get_scheduler`

  ```
  from transformers import get_scheduler
  epochs = 3
  num_training_steps = epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps,
  )
  print(num_training_steps)
  ```

  损失函数：`CrossEntropyLoss`

  ```
  loss_fn = nn.CrossEntropyLoss()
  ```

  

