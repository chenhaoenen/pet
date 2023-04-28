# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/04/28 15:08
# --------------------------------------------
import time
import torch
from datetime import datetime
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter
from src.dataset import AgNewsDataset, collate_fn
from torch.utils.data import DataLoader


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


def setup_training(config):
    set_seed(config.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate} layers={config.transformer_num_hidden_layers}'
    writer = SummaryWriter(log_dir='./data/tensorboard/' + time_stamp, comment=comment)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    return config, device, writer

def prepare_data_loader(config, num_workers=1):
    train_dataset = AgNewsDataset(config.train_data_path, config.model_path, config.pet_repetitions, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)
    return train_data_iter

def trainer():
    config = Config()
    data_iter = prepare_data_loader(config)
    for w in data_iter:
        input_ids, token_type_ids, attention_mask, mlm_labels, label = w
        print(mlm_labels)
        print(label)
        break

if __name__ == '__main__':
    trainer()
