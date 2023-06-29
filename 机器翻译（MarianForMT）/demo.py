import os
import torch
import numpy as np

from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from transformers import AutoConfig,  AutoModelForSeq2SeqLM,AutoTokenizer
from pre_process_data import data_pre
from utils import arg
from model.model import MarianForMT
from utils.log import logger
args = arg.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(args.device)
        outputs = model(**batch_data)
        '''
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))
        '''
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def dev_loop(dataloader, model):
    preds, labels = [], []
    # bleu = BLEU()
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(args.device)
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
            ).cpu().numpy()
            label_tokens = batch_data["labels"].cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            preds += [pred.strip() for pred in decoded_preds]
            labels += [[label.strip()] for label in decoded_labels]
    # return bleu.corpus_score(preds, labels).score


def train(train_data, dev_data, model):
    t_total = len(train_data) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_data)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    # best_bleu = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(train_data, model, optimizer, lr_scheduler, epoch, total_loss)
        dev_loop(dev_data, model)
        # logger.info(f'Dev: BLEU - {dev_bleu:0.4f}')
        # if dev_bleu > best_bleu:
        #     best_bleu = dev_bleu
        #     logger.info(f'saving new weights to {args.output_dir}...\n')
        #     save_weight = f'epoch_{epoch+1}_dev_bleu_{dev_bleu:0.4f}_weights.bin'
        #     torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")


def train(train_data, dev_data, model):
    t_total = len(train_data) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_data)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_bleu = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(train_data, model, optimizer, lr_scheduler, epoch, total_loss)
        dev_loop(dev_data, model)
        # logger.info(f'Dev: BLEU - {dev_bleu:0.4f}')
        # if dev_bleu > best_bleu:
        #     best_bleu = dev_bleu
        #     logger.info(f'saving new weights to {args.output_dir}...\n')
        #     save_weight = f'epoch_{epoch+1}_dev_bleu_{dev_bleu:0.4f}_weights.bin'
        #     torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")



def predict(sentence, model, tokenizer):
    inputs = tokenizer(
        sentence,
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    inputs = inputs.to(args.device)
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_target_length,
        ).cpu().numpy()
    decoded_preds = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return decoded_preds


def main():
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    '''加载模型'''
    # config = AutoConfig.from_pretrained(args.model_checkpoint)
    # model =  MarianForMT.from_pretrained(args.model_checkpoint, config=config).to(args.device)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    # Training
    if  args.do_train:
        '''加载数据'''
        train_data = data_pre.yield_data(args.train_file, shuffle=True)
        dev_data = data_pre.yield_data(args.train_file, shuffle=False)
        '''训练数据'''
        train(train_data, dev_data, model)
    '''测试数据'''
    sentence = '不'
    answer = predict(sentence, model, tokenizer)
    print(answer)


if __name__ == '__main__':
   main()