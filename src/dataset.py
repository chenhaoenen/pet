# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/04/28 15:19
# --------------------------------------------
import csv

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer



class AgNewsDataset(Dataset):
    VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech"]
    }
    VERBALIZER_INDEX = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3
    }

    def __init__(self, data_path, model_path, pattern_id, max_length):
        self.examples = []
        self.tokenizer = BertTokenizer.from_pretrained(model_path) #加载tokenizer
        self.mask = self.tokenizer.mask_token # 获取tokenizer中mask标签</mask>
        self.mask_id = self.tokenizer.mask_token_id # 词表中</mask>对应的id

        self.max_length = max_length
        self.pattern_id = pattern_id
        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                if label not in self.VERBALIZER_INDEX: continue
                example = [text_a, text_b, self.VERBALIZER_INDEX[label]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def encode(self, text_a, text_b):

        if self.pattern_id == 0:
            prompt_text =  [self.mask, ':', text_a, text_b]
        elif self.pattern_id == 1:
            prompt_text =  [self.mask, 'News:', text_a, text_b]
        elif self.pattern_id == 2:
            prompt_text =  [text_a, '(', self.mask, ')', text_b]
        elif self.pattern_id == 3:
            prompt_text = [text_a, text_b, '(', self.mask, ')']
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text_a, text_b]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text_a, text_b]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 return_tensors='pt')
        return feature


    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def get_verbalizer(self, label):
        return self.VERBALIZER[label]

    def __getitem__(self, idx):
        text_a, text_b, label = self.examples[idx]
        feature = self.encode(text_a, text_b)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        input_ids = feature.input_ids.tolist()[0]
        mlm_labels = self.get_mlm_labels(input_ids)
        return input_ids, token_type_ids, attention_mask, mlm_labels, label


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, mlm_labels, labels = zip(*batch)
    mlm_labels = torch.stack([torch.Tensor(mlm_label).long() for mlm_label in mlm_labels])
    labels = torch.stack([torch.Tensor([label]).long() for label in labels])

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels









