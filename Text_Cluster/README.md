#  Clustering_Bert_Auto_Enconding

*Author : wangwenju；QQ: 2413887576； 邮箱：wangwenju_wangs@163.com*

***

**specific task: 短文本聚类**

***

短文本的词汇数量少，提取其中的语义特征信息困难，利用传统模型VSM向量化表示，容易得到高维稀疏向量。词的稀疏表示缺少语义相关性，导致下游聚类任务中，准确率低下，容易受噪声干扰等问题。本项目利用预训练模型 BERT作为文本表示的初始化方法，利用自动编码器对文本表示向量进行自训练以提取高阶特征，将得到的特征提取器和聚类模型进行联合训练，Joint优化特征提取模块和聚类模块，提高聚类模型的准确度和鲁棒性。

- ### **Data Loader** 

   固定模板，`collate_fn`自定义模型所接受数据的格式，参考代码如下。

```python
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
                text.append({'text': l['text']})
        return text
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sent = self.data[index]['text']
        features = self.tokenizer(sent, truncation=True,
                                  max_length=args.max_length,
                                  padding='max_length')

        input_ids = features["input_ids"]
        token_type_ids = features["token_type_ids"]
        attention_mask = features["attention_mask"]
        one_data = {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask}
        return one_data
def collate_fn(batch_data):
    input_ids =  [item["input_ids"] for item in batch_data]
    token_type_ids = [item["token_type_ids"] for item in batch_data]
    attention_mask = [item["attention_mask"] for item in batch_data]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids,
            "attention_mask": attention_mask}
```

此外，也可以直接返回`tensor`格式，只需在`tokenizer`中加入参数`return_tensor =='pt'`,注意返回维度后需要用`squeeze`处理。

- ### **Model**

  - ***自编码任务***：

    自编码网络通过编码器提取高维特征并降维处理 ，然后对输入进行重构。目的是利用神经网络拟合一个恒等函数，提高编码器的特征提取能力，重构过程采用均方误差作为损失函数。  

  ```
  class Auto_encoder(nn.Module):
        def __init__(self,input_size, args):
            super(Auto_encoder,self).__init__()
            self.hidden = args.hidden
            self.num_dimension = args.num_dimension
            ''' 编码器 '''
            self.encoder = nn.Sequential(
                   nn.Linear(input_size, self.hidden),
                   nn.ReLU(inplace=True),
                   nn.Linear(self.hidden,self.num_dimension),
                   nn.Sigmoid())
            '''解码器'''
            self.decoder = nn.Sequential(
                 nn.Linear(self.num_dimension,self.hidden),
                 nn.ReLU(inplace=True),
                 nn.Linear(self.hidden,input_size))
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
  ```

  - ***K_means聚类任务***

    1.初始化 K-Means 聚类算法的簇心 $u_j$, 然后计算每个样本点 i 属于簇 j 的概率 $q_{ij}$ ，得到样本点的概率分布 Q 。  
    $$
    q_{ij} = \frac{(1+||z_i-u_j||^2/v)^{-\frac{v+1}{2}}}{\sum_{j}(1+||z_i-u_j||^2)^{-\frac{v+1}{2}}}
    $$
    其中，$z_i$ 表示样本点的特征向量，$u_j$ 表示簇心向量，v是 t 分布的自由度，取值为1。  

     2.自监督学习的目标分布采用辅助目标分布 P ,P 更近似于原数据分布，作为自训练阶段的辅助目标函数。
    $$
    p_{ij} = \frac{q_{ij}^2/{\sum_{i}q_{ij}}}{\sum_j{q_{ij}^2/\sum_i{q_{ij}}}}
    $$
    其中，$q_{ij}$ 表示样本 i 属于簇心 j 的估计概率 .

    3. 联合训练编码器Encoder和聚类网络 K-Means。需要度量两个分布 Q 和 P 之间的差异，采用KL 散度（KL-divergence）作为损失函数训练模型。

    ```
    import torch.nn as nn
    from transformers import BertPreTrainedModel, BertModel
    from data_pre_process.tools import tokenizer
    import torch.nn.functional as F
    import torch
    import math
    class Auto_encoder(nn.Module):
          def __init__(self,input_size, args):
              super(Auto_encoder,self).__init__()
              self.hidden = args.hidden
              self.num_dimension = args.num_dimension
              ''' 编码器 '''
              self.encoder = nn.Sequential(
                     nn.Linear(input_size, self.hidden),
                     nn.ReLU(inplace=True),
                     nn.Linear(self.hidden,self.num_dimension),
                     nn.Sigmoid())
              '''解码器'''
              self.decoder = nn.Sequential(
                   nn.Linear(self.num_dimension,self.hidden),
                   nn.ReLU(inplace=True),
                   nn.Linear(self.hidden,input_size))
          def forward(self, x):
              encoded = self.encoder(x)
              decoded = self.decoder(encoded)
              return encoded, decoded
    
    
    class Bert_cluster(BertPreTrainedModel):
        def __init__(self, config, params):
            super().__init__(config)
            self.config = config
            self.hidden_size = config.hidden_size
            self.rel_num = params.rel_num
            self.batch_size = params.batch_size
            self.num_dimension = params.num_dimension
            '''加载模型'''
            self.bert = BertModel(config= self.config)
            self.bert.resize_token_embeddings(len(tokenizer))
            self.auto_encoder = Auto_encoder(self.hidden_size,params)
            # DEC_cluster
            '''随机 初始化K_MEANS 质心向量'''
            self.rel_embedding = nn.Embedding(params.rel_num,params.num_dimension)
            self.init_weights()
    
        def forward(self, input_ids,attention_mask,token_type_ids):
            output = self.bert(input_ids=input_ids,attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
            cls_hidden_state = output.pooler_output
            # print(cls_hidden_state.size())
            encoded, decoded = self.auto_encoder(cls_hidden_state)
            '''初始化质心位置'''
            index = torch.arange(0, self.rel_num, device=input_ids.device).repeat(self.batch_size,1)
            centorid = self.rel_embedding(index)
    
            ''' 1:计算近似估计软分布Q; 这里做修改: 当前 第i个 sentense 样本看作 quary, 每个簇质心的位置可以看作 keys
            求解 quary 和 不同 keys 之间距离的相似度，采用attention 机制代替原有的方法'''
            Q_distribute = self.compute_Q(encoded,centorid)
            '''2:计算辅助目标分布P'''
            P_distribute = self.compute_P(Q_distribute)
    
            return cls_hidden_state, decoded, Q_distribute, P_distribute
    
        def compute_Q(self,query,key):
            scores = torch.matmul(torch.unsqueeze(query,dim = 1), key.permute(0,2,1)) / math.sqrt(self.num_dimension)
            scores = F.softmax(scores.squeeze(), dim=-1)
            return scores
        '''DEC原始paper实现方式'''
        def compute_Q_origin_paper(self,encoded,centorid, v = 1):
            temp = torch.sub(encoded.unsqueeze(dim = 1).expand(-1,self.rel_num,-1),centorid)
            numerator = torch.pow(torch.add(1,torch.norm(temp, dim = -1 ) / v), -(v+1)/2)
            denominator = torch.sum(numerator, dim=1)
            Q_distribute = torch.div(numerator, denominator.unsqueeze(dim=-1))
            '''计算P和Q的KL散度，做softmax处理，防止出现负数'''
            Q_distribute = F.softmax(Q_distribute, dim=-1)
            return Q_distribute
    
        def compute_P(self, Q_distribute):
            Q_2_distribute = torch.pow(Q_distribute, 2)
            F_i_j = torch.sum(Q_distribute, dim = 0)
            # print(Q_2_distribute.size())
            # print(F_i_j.size())
            numerator = torch.div(Q_2_distribute,F_i_j.repeat(self.batch_size,1))
            denominator =  torch.sum(numerator, dim = 1)
            P_distribute = torch.div(numerator,denominator.unsqueeze(dim = -1))
            '''计算P和Q的KL散度，做softmax处理，防止出现负数'''
            P_distribute = F.softmax(P_distribute, dim=-1)
            return P_distribute
    ```

    本项目计算软分布Q，也可以采用注意力机制，代码如上图所示。

  - #### ***联合训练***

​              		

```python
for item in train_data:
            input_ids, input_seg, input_mask = item["input_ids"].to(device),  \
                                             item["token_type_ids"].to(device), \                                          item["attention_mask"].to(device)
            logits_enconder, logit_deconder,  Q_distribute, P_distribute = model(input_ids,input_mask,input_seg)
            loss1 = F.mse_loss(logits_enconder, logit_deconder)
            loss2 = KL_criterion(Q_distribute.log(), P_distribute)
            loss = loss1 + loss2
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)  # 梯度裁剪
            schedule.step()
```

+ <u>**参考链接**</u>

  + [huggingface 官网教程](https://huggingface.co/course)

  + [融合BERT和自编码网络的短文本聚类研究](https://xueshu.baidu.com/usercenter/paper/show?paperid=1w3j00q0ex2y0rx0635y0xb0pr422570&site=xueshu_se&hitarticle=1)

    
