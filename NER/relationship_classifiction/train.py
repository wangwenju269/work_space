import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
from utils.arguments_parse import args
from data_preprocessing import data_prepro
from model.model import bertMRC
from model.loss_function import cross_entropy_loss,binary_cross_entropy,multilabel_cross_entropy
from model.loss_function import span_loss
from model.loss_function import focal_loss
from model.metrics import metrics
from utils.logger import logger

device = torch.device('cuda')
sentences=data_prepro.load_data(args.train_path)
print('总共含有句子数量：',len(sentences))
train_data_length=len(sentences)


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_epoch, decay_epoch, min_lr_rate=1e-8):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.epoch_step = train_data_length / args.batch_size
        self.warm_up_steps = self.epoch_step * warm_up_epoch
        self.decay_steps = self.epoch_step * decay_epoch
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0
        self.all_steps = args.epoch*(train_data_length/args.batch_size)

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
            rate = self.init_rate
        else:
            rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (self.all_steps-self.decay_steps))) * self.init_rate
            if rate < self.min_lr_rate:
                rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

def train():
    train_data = data_prepro.yield_data(args.train_path)
    test_data = data_prepro.yield_data(args.dev_path)

    model = bertMRC(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
    #model.load_state_dict(torch.load(args.checkpoints))
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)

    schedule = WarmUp_LinearDecay(
                optimizer = optimizer, 
                init_rate = args.learning_rate,
                 warm_up_epoch = args.warm_up_epoch,
                decay_epoch = args.decay_epoch
            )
    
    loss_func = cross_entropy_loss.cross_entropy().to(device)
    acc_func = metrics.metrics_func().to(device)

    step=0
    for epoch in range(args.epoch):
        for item in train_data:
            step+=1
            input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
            labels,flag = item["labels"],item["flag"]
    
            optimizer.zero_grad()
            logits = model( 
                input_ids=input_ids.to(device), 
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                flag=flag.to(device),
                is_training=True
            )

            loss= loss_func(logits,labels.to(device))
            loss = loss.float().mean().type_as(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            schedule.step()

            if step%100 == 0:
                logits=torch.nn.functional.softmax(logits, dim=-1)
                recall, precise, f1=acc_func(logits,labels.to(device))

                logger.info('epoch %d, step %d, loss %.4f, recall %.4f, precise %.4f, f1 %.4f'% (
                    epoch,step,loss,recall, precise, f1))

        with torch.no_grad():

            recall, precise, f1=0,0,0
            count=0

            for item in test_data:
                count+=1
                input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
                labels,flag = item["labels"],item["flag"]
        
                optimizer.zero_grad()
                logits = model( 
                    input_ids=input_ids.to(device), 
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device),
                    flag=flag.to(device),
                    is_training=False
                )
                tmp_recall, tmp_precise, tmp_f1=acc_func(logits,labels.to(device))
                f1+=tmp_f1
                recall+=tmp_recall
                precise+=tmp_precise

            f1/=count
            recall/=count
            precise/=count

            logger.info('-----eval----')
            logger.info('epoch %d, step %d, loss %.4f, recall %.4f, precise %.4f, f1 %.4f'% (
                    epoch,step,loss,recall, precise, f1))
            logger.info('-----eval----')
            torch.save(model.state_dict(), f=args.checkpoints)
            logger.info('-----save the best model----')


if __name__=='__main__':
    train()
