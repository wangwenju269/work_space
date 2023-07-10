## 关系抽取（联合解码方法）

+ PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction

- 来源：ACL 2021
- 论文地址：https://arxiv.org/pdf/2106.09895
- 开源代码：[https://github.com/hy-struggle/PRGC](https://hub.fastgit.xyz/hy-struggle/PRGC)

### Abstract： 

​       从非结构化文本中联合抽取实体和关系是信息抽取的一个关键问题。一些方法取得了良好的性能，但仍存在一些固有的局限性，如关系预测的冗余性、基于区间的提取泛化能力差和效率低下。文章从一个新的角度将任务分解为关系判断、实体提取和主客体对齐三个子任务，并提出了一个基于潜在关系和全局对应的联合三胞组提取框架。

- **three subtasks**
- Relation Judgement：设计了一个预测潜在关系的组件，将实体的提取限制在预测的关系子集，而不是所有的关系；
- Entity Extraction：应用特定于关系的序列标注组件处理主体与客体之间的重叠问题；
- Subject-object Alignment：全局对应组件是将主体与客体对齐，以低复杂度的方式组合成一个三元组；

### **文章工作**

​        关系抽取可以分为流水线和联合提取的方法，文章采用端到端联合提取三元组，主要解决问题如下：

​       \- span-based方法：仅仅关注于实体的首尾，泛化性能差；

​        \- Casrel方法：首先提取主体，每种关系分别看做映射函数，匹配相对应的客体，很容易造成关系冗余，效率低下；

本文将关系提取任务分为三个组件：Potential Relation Prediction从所有冗余关系中，预测出候选关系；Entity Extraction主要采用 Relation-Specific Sequence Tagging 的方法解决主体客体的 overlap问题；Global Correspondence matrix 确定 specific实体关系对是否有效；

### **模型架构：**

![img](https://pic1.zhimg.com/80/v2-aa68e306240e86dcc8574f6e5f95a8ef_720w.png?source=d16d100b)



编辑切换为居中

添加图片注释，不超过 140 字（可选）

*Figure 1: The overall structure of PRGC. Given a sentence S, PRGC predicts a subset of potential relations Rpot and a global correspondence M which indicates the alignment between subjects and objects. Then for each potential relation, a relation-specific sentence representation is constructed for sequence tagging. Finally we enumerate all possible subject-object pairs and get four candidate triples for this particular example, but only two triples are left (marked red) after applying the constraint of global correspondence.*

### **PRGC编码器**

句子的输入定义为  ,关系三元组输出  ，公式表示如下。  和  分别是实体和关系的集合；

 

- **Relation Judgement** ：给定句子  ，预测包含的潜在关系，输出形式如下，其中  是潜在关系集合的大小。  
- **Entity Extraction:** 给定句子  和预测候选关系  ,BIO标注每个token，标记为  ;输出形式记做如下：  
- **Subject-object Alignment :**对于给定的句子S，该子任务预测主体和客体start token间的相对应分数。M表示全局对应矩阵，输出形式记做如下；  
- 用bert作为预训练编码器，编码器输出为  , d是嵌入向量维度，n为句子token数目；

###  **PRGC 解码器**

- **Potential Relation Prediction：**对于给定的句子，我们首先预测句子中可能存在的潜在关系子集，然后对这些潜在关系进行实体提取。给定一个含有n个token的句子隐向量  ，对潜在关系分类：  

> 这里，  为待训练参数，   , 建模为多标签的二值分类任务，当  概率超过某个阈值  时，为预测关系；

- **Relation-Specific Sequence Tagging**: 执行两个序列标记操作来提取主体和客体，提取主体和客体的原因是为了处理特殊的重叠模式Subject Object Overlap (SOO)；

 

> 这里  是第j 个关系表示，从可训练参数矩阵  中选取，  是所有关系的集合；  是可训练参数，集合的标注{B，I，O};

- **Global Correspondence：**使用全局对应矩阵来确定主体和客体的正确配对，首先，列举了所有可能的主体客体对，然后在全局矩阵中对每对实体pairs进行相应的评分，如果值超过某个阈值  ，则保留该值，否则过滤掉。correspondence 矩阵记作为  ,矩阵中元素计算如下： ；待确定的参数为  ；

### **损失函数**

联合训练模型，在训练期间优化损失目标函数，共享PRGC编码器的参数。

 

这里，  是所有关系的集合，  是预测关系的集合；总体损失是三者相加；

### **对比实验**

![img](https://pic1.zhimg.com/80/v2-cdffd97afe8ee8ed4ee4ba7d1bfaff83_720w.png?source=d16d100b)



编辑切换为居中

添加图片注释，不超过 140 字（可选）

### **结论：**

​	    	提出了一种全新的视角，提出了一种基于预测关系和全局对应的联合关系抽取框架，极大地缓解了关系判断冗余、span_base抽取泛化能力差和主客体对齐效率低的问题。实验结果表明，我们的模型在公共数据集上达到SOTA，能处理许多复杂的场景。



## Requirements

The main requirements are:

  - python==3.7.9
  - pytorch==1.6.0
  - transformers==3.2.0
  - tqdm

## Datasets

- [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT) and [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)(following [CasRel](https://github.com/weizhepei/CasRel))
- [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)(following [CopyRE](https://github.com/xiangrongzeng/copy_re))
- [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)(following [ETL-span](https://github.com/yubowen-ph/JointER))



##  代码实现

+ 随机种子

  ```python
   # Set the random seed for reproducible experiments
  random.seed(seed)
  torch.manual_seed(seed)
  if n_gpu > 0:
     torch.cuda.manual_seed_all(seed)
  ```

+ 多卡`GPU` 设置

  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count() # 读取GPU的数量
  """注意将model搬到GPU时，操作:"""
  model.to(params.device)
  # parallel model
  if params.n_gpu > 1 and args.multi_gpu:
      model = torch.nn.DataParallel(model)
  
  ```

+ 数据处理

  本项目`dataloader` 非常经典，很值得学习，大致思路：`dataloader` <---- `Feature_dataset` <----`convert_examples_to_features  ` <--- ` read_examples`

  + ` read_examples`

    ```python
    class InputExample(object):
        """ 定义储存数据结构的类，a single set of samples of data
        """
        def __init__(self, text, en_pair_list, re_list, rel2ens):
            self.text = text
            self.en_pair_list = en_pair_list
            self.re_list = re_list
            self.rel2ens = rel2ens
    ## 接下来将数据转化上述类的形式储存。
    def read_examples(data_dir, data_sign, rel2idx):
        """ load data to InputExamples
        """
        examples = []
        with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
            data = json.load(f)
            for sample in data:
                text = sample['text']
                rel2ens = defaultdict(list)
                en_pair_list = []
                re_list = []
                for triple in sample['triple_list']:
                    en_pair_list.append([triple[0], triple[-1]])
                    re_list.append(rel2idx[triple[1]])
                    rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
                example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
                examples.append(example)
        print("InputExamples:", len(examples))
        return examples
    ```

  + `convert_examples_to_features`

    ```python
    ### 加速计算，调用 python 多线程计算方法
    def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
        """convert examples to features.
        """
        max_text_len = params.max_seq_length
        # multi-process
        with Pool(10) as p:
            convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,data_sign=data_sign, ex_params=ex_params)
            features = p.map(func=convert_func, iterable=examples)
        return list(chain(*features))
    """
    代码理解：
    1. `convert`是自定义转化函数，所接受的参数主要有 `max_text_len`, `tokenizer`, `rel2idx`,`data_sign`,`ex_params`
    2. map 操作:  features = p.map(func=convert_func, iterable=examples)
    3. chain 操作：list(chain(*features)),收集所有features,并进行拉直。 
    """
    ```

   + `convert`

     ```python
     def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
         """convert function
         """
         text_tokens = tokenizer.tokenize(example.text)
         # cut off
         if len(text_tokens) > max_text_len:
             text_tokens = text_tokens[:max_text_len
         # token to id
         input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
         attention_mask = [1] * len(input_ids)
         # zero-padding up to the sequence length
         if len(input_ids) < max_text_len:
             pad_len = max_text_len - len(input_ids)
             # token_pad_id=0
             input_ids += [0] * pad_len
             attention_mask += [0] * pad_len
     """
     代码理解：
         1.上述提供一种 tokenizer 分词方式，用下面的写法难道不香吗？  
          features = tokenizer(    text,
                                   padding = 'max_length',
                                   max_length = args.max_seq_length,
                                   truncation = True,
                                   return_tensors='pt')
         答：在处理英文数据上，简单使用tokenizer(text),很大程度上计算资源的浪费，args.max_seq_length是基于char字符级别的最大字符长度值，而tokenizer 分词的方法基于BPE,Wordpiece，subword等，采用tokenizer(text)写法，会造成padding过度长。    
     """                          
                              
     ```

     + `convert`

​               训练集和验证集特征表示方式不一致，怎么怎么实现共享一个dataset呢？，作者给出一种新的方法。	

```python
class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """
    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag
```

想法很简单：统一训练集和验证集的输入形式，当然转化类实例表示呀，在实例上构造`Dataset`迭代器，真是牛。

```
class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """
    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]
```

这又引入一个新问题：实例上包含训练集和验证集合的所有属性，在不同`train`，`dev` 阶段，怎么提取所想要的输入形式呢，别急，作者又引入`collate_fn_train`,`collate_fn_dev`。

上述数据处理的思想真的值得学习，反复看，收获颇丰。



+ 模型环节

​       也是采用很主流的写法，继承`BertPreTrainedModel`   `PretrainModel` 基类。

```python
bert_config =BertConfig.from_json_file(os.path.join(params.bert_model_dir,'config.json'))
model = BertForRE.from_pretrained(   config=bert_config,
                                     pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
```

```
class BertForRE(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        # pretrain model
        self.bert = BertModel(config)
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, params.rel_num, params.drop_prob)
        '''
        pass
        '''
        self.rel_embedding = nn.Embedding(params.rel_num, config.hidden_size)
        self.init_weights()
```

​      模型代码与论文里所述的方法一致，具体细节，后续是怎么计算3种损失函数。

+ 1.  关系分类损失：

     ```python
     @staticmethod
     def masked_avgpool(sent, mask):
         mask_ = mask.masked_fill(mask == 0, -1e9).float()
         score = torch.softmax(mask_, -1)
         return torch.matmul(score.unsqueeze(1), sent).squeeze(1)
     
     h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
                 # (bs, rel_num)
     rel_pred = self.rel_judgement(h_k_avg)
     loss_func = nn.BCEWithLogitsLoss(reduction='mean')
     loss_rel = loss_func(rel_pred, rel_tags.float())
     '''与sentences分类一致'''
     ```

  2. 序列标注损失：

     ```python
     # rel_emb 是候选关系的 embedding 表示，与 token representation 拼接
     decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
     output_sub = self.sequence_tagging_sub(decode_input)
     output_obj = self.sequence_tagging_obj(decode_input)
     ### 拉长，计算主体和客体标注的损失loss
     attention_mask = attention_mask.view(-1)
     loss_func = nn.CrossEntropyLoss(reduction='none')
     loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                           seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
     loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                           seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
     loss_seq = (loss_seq_sub + loss_seq_obj) / 2
     ```

     3.  `subject`,`object` 对齐操作

        ```python
        corres_pred = corres_pred.view(bs, -1)
                        corres_mask = corres_mask.view(bs, -1)
                        corres_tags = corres_tags.view(bs, -1)
                        loss_func = nn.BCEWithLogitsLoss(reduction='none')
                        loss_matrix = (loss_func(corres_pred,
                                                 corres_tags.float()) * corres_mask).sum() / corres_mask.sum()                           
        ```

综合以上，该项目的数据处理，模型结构设计主要代码，是值得学习的一份代码。
