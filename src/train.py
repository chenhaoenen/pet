# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/04/28 15:08
# --------------------------------------------
import time
import torch
from datetime import datetime
from transformers import set_seed
from transformers.modeling_bert import BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD
from src.dataset import AgNewsDataset, collate_fn



class Config():
    def __init__(self):
        self.batch_size = 2
        self.epochs = 50
        self.seed = 1234
        self.learning_rate = 1.5e-5
        self.log_freq = 100
        self.eval_freq = 500
        self.max_seq_length = 128
        self.train_data_path = '../data/ag_news_csv/train.csv'
        self.eval_data_path = '../data/ag_news_csv/test.csv'
        self.model_path = '../model/bert-base-uncased'
        self.output_dir = '../output'

        #pet config
        self.pet_repetitions = 3
        self.pattern_ids = 1
        self.pet_batch_size = 4
        self.max_num_verbalizers = 1


def setup_training(config):
    set_seed(config.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate}'
    writer = SummaryWriter(log_dir='./data/tensorboard/' + time_stamp, comment=comment)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    return config, device, writer

def prepare_data_loader(config, num_workers=1):
    train_dataset = AgNewsDataset(config.train_data_path, config.model_path, config.pet_repetitions, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)
    return train_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len

def prepare_model_and_optimizer(config, device):
    model = BertForSequenceClassification.from_pretrained(config.model_path) #torch.Size([B, max_length, 30522])
    model.to(device)
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    return model, criterion, optimizer

def trainer():
    config = Config()
    config, device, writer = setup_training(config)
    train_iter, m2c_tensor, filler_len = prepare_data_loader(config)
    model, criterion, optimizer = prepare_model_and_optimizer(config, m2c_tensor, filler_len, device)

    total_step = config.epochs * len(train_iter)

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
            input_ids, token_type_ids, attention_mask, mlm_labels, labels = [w.to(device) for w in batch]

            optimizer.zero_grad()
            logit = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(logit, label)

            # print(loss)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            avg_loss += loss

            # tensorboard
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('avg_loss', avg_loss / step, step)

if __name__ == '__main__':
    trainer()
