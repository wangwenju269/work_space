# BERT_MRC

### 概述

基于bert模型，采用首尾指针标注的方法，抽取文本中主体subject 和客体object,该项目主要介绍数据标注-模型训练-模型推理等环节 。 

### 环境要求

```
pytorch >=1.6.0
transformers>=3.4.0
```
### 数据来源

```
class binary_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.BCELoss()

    def forward(self,y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_pred = y_pred.view(size=(-1,))
        y_true = y_true.view(size=(-1,))
        loss = self.loss_func(input=y_pred, target=y_true)
        return loss
```

+ 数据形式样例：{`"text"`: "郭造卿（1532—1593），字建初，号海岳，福建福清县化南里人（今福清市人），郭遇卿之弟，郭造卿少年的时候就很有名气，曾游学吴越", `"spo_list"`: [{`"predicate"`: "号", `"object_type"`: {"@value": "Text"}, `"subject_type"`: "历史人物", `"object"`: {"@value": "海岳"}, `"subject"`: "郭造卿"}]}
+ 数据说明：`"predicate"`：主体与客体间关系，`"object_type"`：客体类型，`"subject_type"`:主体类型， `"subject"`、`"object"`: 分别表示主体，客体；



### 参数解析

```
# 参数解析;
import argparse
parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="data/train.json",help="train file")
args = parser.parse_args()
# 也可以通过下述方法
@dataclass
class DataArguments:
      train_path:str = field(
      default = "--train_path",
      metadata = {"help" : 'train file'}
      )
parser = PdArgumentParser((DataArguments))      
args = parser.parse_args_into_dataclasses()
# 提供以上两种参数解析化的方法
```

### 数据处理

+ 模型训练需要构造迭代器，迭代器基础框架：包含`__init__`、 `__len__`和`__getitem__`属性。

```
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        item = self.data[index]
        return item
```

+ 载入数据，处理预数据模块

  ```
  class MyDataset(Dataset):
      def __init__(self, data_path_files):
          self.data = self.load_data(data_path_files)
      def __len__(self):
          return len(self.data)
      def __getitem__(self, index):
          item = self.data[index]
          ######## encode / process your item #############
          return item
      def load_data(self,file_path):
          with open(file_path, 'r', encoding='utf8') as f:
              lines = f.readlines()
              result=[]
              for line in tqdm(lines):
                  data = json.loads(line)
                  text = data['text']
                  ########### 自定义处理 target #################
                  result.append({'text':text,'spo_list':target})
              return result
  ```

  + 文本分词：token2ids

    ```python
    def get_tokenizer():
        """添加特殊中文字符和未使用的token【unused1】"""
        added_token=['[unused'+str(i+1)+']' for i in range(55)]
        # path=args.pretrained_model_path
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, additional_special_tokens=added_token)
        special_tokens_dict = {'additional_special_tokens':['”','“']}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer
        
    tokenizer=get_tokenizer()
    # 注意在vocab.txt文件新增未知token，需要加载model时,resize_token_embedding
    # self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
    # self.roberta_encoder.resize_token_embeddings(len(tokenizer)
    ```

+  编码、label 首尾位置对齐操作

  ```python
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
          s_startlabel[s_start_idx] = 1
          s_endlabel[s_end_idx] = 1
          o_encode = tokenizer.encode(spo[2])
          o_start_idx = tools.search(o_encode[1:-1], encode_sent)
          o_end_idx = o_start_idx + len(o_encode[1:-1]) - 1
          o_startlabel[o_start_idx] = 1
          o_endlabel[o_end_idx] = 1
      return encode_sent, token_type_ids, attention_mask, s_startlabel,s_endlabel,o_startlabel,o_endlabel
  ```

  + 构建模型框架

    ```python
    class bertMRC(nn.Module):
        def __init__(self, pre_train_dir: str, dropout_rate: float):
            super().__init__()
            self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
            self.roberta_encoder.resize_token_embeddings(len(tokenizer))
            self.s_startlayer = torch.nn.Linear(in_features=768, out_features=1)
            self.s_endlayer = torch.nn.Linear(in_features=768, out_features=1)
            self.o_startlayer = torch.nn.Linear(in_features=768, out_features=1)
            self.o_endlayer = torch.nn.Linear(in_features=768, out_features=1)
            self.sigmoid=nn.Sigmoid()
    
        def forward(self, input_ids, input_mask, input_seg, is_training=False):
            bert_output = self.roberta_encoder(input_ids=input_ids,
                                               attention_mask=input_mask,
                                               token_type_ids=input_seg)  # (bsz, seq, dim)
            encoder_rep = bert_output[0]
            s_startlogits = self.sigmoid(self.s_startlayer(encoder_rep))
            s_endlogits   = self.sigmoid(self.s_endlayer(encoder_rep))
            o_startlogits=self.sigmoid(self.o_startlayer(encoder_rep))
            o_endlogits=self.sigmoid(self.o_endlayer(encoder_rep))
            if is_training:
                return s_startlogits,s_endlogits,o_startlogits,o_endlogits
            else:
                return s_startlogits,s_endlogits,o_startlogits,o_endlogits
    ```

    + 另一种参考写法

      ```python
      config = BertConfig.from_pretrained(pretrain_model_path )
      model = bertMRC.from_pretrained(            pretrain_model_path,
                                                  config=config,
                           )
      # bertMRC 继承 `BertPretrainModel` 类  
      ```

    + 损失函数：`multi_sigmod`的 sequence 二分类损失

      ```
      class binary_cross_entropy(nn.Module):
          def __init__(self):
              super().__init__()
              self.loss_func = torch.nn.BCELoss()
      
          def forward(self,y_pred, y_true):
              y_pred = y_pred.float()
              y_true = y_true.float()
              y_pred = y_pred.view(size=(-1,))
              y_true = y_true.view(size=(-1,))
              loss = self.loss_func(input=y_pred, target=y_true)
              return loss
      ```

      
