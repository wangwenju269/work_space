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

