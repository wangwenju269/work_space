from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import os
import torch
from torch import nn
from transformers import AdamW, get_scheduler
from transformers import BertConfig, AutoTokenizer
from Precoss_Data import data_loader
from Model.model import BertForPairwiseCLS
from tqdm.auto import tqdm
from Utils.argpaser import parse_args
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")
'''评估指标'''
entropy_loss = nn.CrossEntropyLoss()
def pipeline():
    '''
    1:加载数据
    '''
    train_data = data_loader.yield_data(args.train_file)
    dev_data =  data_loader.yield_data(args.train_file)
    args.train_data_length = len(train_data)
    print(len(dev_data))
    '''
    2:构建模型结构
    '''
    bert_config = BertConfig.from_json_file(os.path.join(args.model_checkpoint, 'config.json'))
    model = BertForPairwiseCLS.from_pretrained(config = bert_config,
                                         pretrained_model_name_or_path = args.model_checkpoint,
                                         args = args,
                                         ignore_mismatched_sizes=True).to(args.device)
    '''
    3:优化器选择
    '''
    t_total = args.train_data_length * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    '''
    4:模型训练
    '''
    # 写日志
    logger.info("***** Running training *****")
    logger.info(f"Num examples - { args.train_data_length}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
         f.write(str(args))

    total_loss, best_acc = 0, 0
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_data, model, optimizer, lr_scheduler, epoch, total_loss)
        valid_acc = dev_loop(args,  dev_data, model)
        if valid_acc > best_acc:
            best_acc = valid_acc
            logger.info(f"best accuracy - {best_acc}")
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.save_weight))
    logger.info("Done!")





def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    model.train()
    for step, (batch_data,batch_label) in enumerate(dataloader, start=1):
        input_ids = batch_data['input_ids'].squeeze().to(args.device)
        token_type_ids = batch_data['token_type_ids'].squeeze().to(args.device)
        attention_mask = batch_data['attention_mask'].squeeze().to(args.device)
        outputs = model(input_ids, attention_mask, token_type_ids)

        label = batch_label.to(args.device)
        loss = entropy_loss(outputs,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def dev_loop(args, dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_label in tqdm(dataloader):
            input_ids = batch_data['input_ids'].squeeze().to(args.device)
            token_type_ids = batch_data['token_type_ids'].squeeze().to(args.device)
            attention_mask = batch_data['attention_mask'].squeeze().to(args.device)
            logits = model(input_ids, attention_mask, token_type_ids)
            label = batch_label.to(args.device)
            predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = label.cpu().numpy()
            correct += (predictions == labels).sum()
    correct /= len(dataloader.dataset)
    return correct



if __name__ == '__main__':
    # if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    pipeline()