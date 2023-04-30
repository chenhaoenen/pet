# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/04/28 15:08
# --------------------------------------------
import os
import time
import torch
from datetime import datetime
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from src.dataset import AgNewsDataset, collate_fn
from src.model import PetModel, PetCriterion
from src.utils import stats_time



class Config():
    def __init__(self):
        self.batch_size = 4
        self.epochs = 50
        self.seed = 1234
        self.log_freq = 100
        self.eval_freq = 500
        self.max_seq_length = 256
        self.train_data_path = '../data/ag_news_csv/train.csv'
        self.eval_data_path = '../data/ag_news_csv/test.csv'
        self.model_path = '../model/bert-base-uncased'
        self.output_dir = '../output'
        self.gradient_accumulation_steps = 32
        self.learning_rate = 5e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1
        self.warmup_steps = 0


        #pet config
        self.pet_repetitions = 3
        self.pattern_ids = 1
        self.pet_batch_size = 4
        self.max_num_verbalizers = 1


def setup_training(config):
    set_seed(config.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate}'
    writer = SummaryWriter(log_dir='./data/tensorboard/' + time_stamp, comment=comment)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    return config, writer

def prepare_data_loader(config, num_workers=1):
    train_dataset = AgNewsDataset(config.train_data_path, config.model_path, config.pet_repetitions, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)
    return train_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len

def prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step):
    model = PetModel(config)
    model.to(config.device)
    criterion = PetCriterion(config, m2c_tensor, filler_len)
    criterion.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_step)
    return model, criterion, optimizer, scheduler

def trainer():
    config = Config()
    config, writer = setup_training(config)
    train_iter, m2c_tensor, filler_len = prepare_data_loader(config)
    total_step = config.epochs * len(train_iter)

    model, criterion, optimizer, scheduler = prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step)

    print(f"{'#' * 41} Config {'#' * 41}")
    for k in list(vars(config).keys()):
        print('{0}: {1}'.format(k, vars(config)[k]))
    print(f'total step: {total_step}')
    print(f'the number of train step: {len(train_iter)}')
    print(f"{'#' * 41} Training {'#' * 41}")

    start = int(time.time())
    step = 0
    avg_loss = 0.0
    for epoch in range(1, config.epochs+1):
        # train
        for i, batch in enumerate(train_iter):
            model.train()
            step += 1
            input_ids, token_type_ids, attention_mask, mlm_labels, labels = [w.to(config.device) for w in batch]

            logit = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(logit, mlm_labels, labels)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            avg_loss += loss.item()
            if step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # log
            if step % config.log_freq == 0:
                end = int(time.time())
                print(f"epochs:{str(epoch) + '/' + str(config.epochs)}, batch:{str(i + 1) + '/' + str(len(train_iter))}, step:{str(step) + '/' + str(total_step)}, cur_loss:{'{:.6f}'.format(loss)}, avg_loss:{'{:.6f}'.format(avg_loss/step)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%m/%d %H:%M:%S', time.localtime())}")


            # tensorboard
            # writer.add_scalar('loss', loss, step)
            # writer.add_scalar('avg_loss', avg_loss / step, step)

if __name__ == '__main__':
    trainer()
