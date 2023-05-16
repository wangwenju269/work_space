from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import os
from transformers import BertConfig
import torch
import torch.nn as nn
from torch import optim
from model.Bert2classify import Bert_classify
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
    1:加载数据
    '''
    train_data = pre_processes.yield_data(args.train_path)
    test_data =  pre_processes.yield_data(args.dev_path)
    args.train_data_length = len(train_data)
    print(len(test_data))
    '''
    2:构建模型结构
    '''
    bert_config = BertConfig.from_json_file(os.path.join(args.pretrained_model_path, 'config.json'))
    model = Bert_classify.from_pretrained(config = bert_config,
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
    entropy_loss = nn.CrossEntropyLoss()
    acc_func = metrics.metrics_func()
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
            labels = item["label"].to(device)
            logits = model(input_ids,input_mask,input_seg)
            print(logits)
            loss = entropy_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)  # 梯度裁剪
            schedule.step()
            if  step % args.step == 0:
                acc = acc_func(logits, labels)
                logger.info('epoch %d, step %d, loss %.4f, acc %.4f' % (epoch, step, loss, acc))

        with torch.no_grad():
            count, acc = 0 ,0
            for item in test_data:
                count += 1
                input_ids, input_seg, input_mask = item["input_ids"].to(device), \
                                                   item["token_type_ids"].to(device), \
                                                   item["attention_mask"].to(device)
                labels = item["label"].to(device)
                optimizer.zero_grad()
                logits = model(input_ids, input_mask, input_seg)
                acc += acc_func(logits, labels)
            acc = acc / count
            logger.info('-----eval----')
            logger.info('epoch %d, step %d, acc %.4f' % (epoch, step, acc))
            if acc > best_acc:
                best_acc = acc
                logger.info('-----eval----')
                torch.save(model.state_dict(), f=args.checkpoints)
                logger.info('-----save the best model----')

if __name__ == '__main__':
    train()
