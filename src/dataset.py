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
    VERBALIZER_INDEX_LABEL = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3
    }
    VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER}

    def __init__(self, data_path, model_path, pattern_id, max_length, max_num_verbalizers):
        self.examples = []
        self.tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载tokenizer
        self.mask = self.tokenizer.mask_token  # 获取tokenizer中mask标签</mask>
        self.mask_id = self.tokenizer.mask_token_id  # 词表中</mask>对应的id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max_num_verbalizers
        self.m2c_tensor = self._buid_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                if label not in self.VERBALIZER_INDEX_LABEL: continue
                example = [text_a, text_b, self.VERBALIZER_INDEX_LABEL[label]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _buid_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 #[len(VERBALIZER_LABEL), max_num_verbalizers]对角线为-1的对角矩阵
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor #[label_size, max_num_verbalizers]

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len #[label_size] 有的label 的verbalize比较多，有的label的verbalize较少

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids


    def encode(self, text_a, text_b):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text_a, text_b]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'News:', text_a, text_b]
        elif self.pattern_id == 2:
            prompt_text = [text_a, '(', self.mask, ')', text_b]
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
        input_ids = input_ids.tolist()[0]
        mlm_labels = self.get_mlm_labels(input_ids)
        return input_ids, token_type_ids, attention_mask, mlm_labels, label


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, mlm_labels, labels = zip(*batch)
    mlm_labels = torch.stack([torch.Tensor(mlm_label).long() for mlm_label in mlm_labels])
    labels = torch.stack([torch.Tensor([label]).long() for label in labels])

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels
