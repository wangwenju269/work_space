from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import os
from transformers import BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model.Bert2cluster import Bert_cluster
from utils.arguments_parse import args
from utils.logger import logger
from data_pre_process import pre_processes
from metrics import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_epoch, decay_epoch, min_lr_rate=1e-8):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.epoch_step = args.train_data_length / args.batch_size
        self.warm_up_steps = self.epoch_step * warm_up_epoch
        self.decay_steps = self.epoch_step * decay_epoch
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0
        self.all_steps = args.epoch*(args.train_data_length/args.batch_size)

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
            rate = self.init_rate
        else:
            rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (self.all_steps-self.decay_steps))) * self.init_rate
            if rate < self.min_lr_rate:
               rate = self.min_lr_rates
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

def train():
    '''
    1:加载数据: 根据训练集的文本数据做聚类。
    '''
    train_data = pre_processes.yield_data(args.train_path)
    args.train_data_length = len(train_data)
    print(len(train_data))
    '''
    2:构建模型结构
    '''
    bert_config = BertConfig.from_json_file(os.path.join(args.pretrained_model_path, 'config.json'))
    model = Bert_cluster.from_pretrained(config = bert_config,
                                         pretrained_model_name_or_path = args.pretrained_model_path,
                                         params = args,
                                         ignore_mismatched_sizes=True).to(device)
    '''
    3:优化器选择
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
    optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)
    schedule = WarmUp_LinearDecay(
        optimizer=optimizer,
        init_rate=args.learning_rate,
        warm_up_epoch=args.warm_up_epoch,
        decay_epoch=args.decay_epoch)
    '''
    4:评估指标
    '''
    KL_criterion = torch.nn.KLDivLoss(reduction ='sum')
    '''
    5:训练模型参数
    '''
    # patience stage
    best_acc = 0.0
    step = 0
    for epoch in range(args.epoch):
        model.train()
        for item in train_data:
            step += 1
            input_ids, input_seg, input_mask = item["input_ids"].to(device),  \
                                               item["token_type_ids"].to(device), \
                                               item["attention_mask"].to(device)
            logits_enconder, logit_deconder,  Q_distribute, P_distribute = model(input_ids,input_mask,input_seg)
            loss1 = F.mse_loss(logits_enconder, logit_deconder)
            loss2 = KL_criterion(Q_distribute.log(), P_distribute)
            loss = loss1 + loss2
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)  # 梯度裁剪
            schedule.step()


if __name__ == '__main__':
    train()
