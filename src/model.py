import torch
import torch.nn as nn
from transformers.models.bert import BertForTokenClassification, BertForMaskedLM
import numpy as np


class PetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained(config.model_path)

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return out.logits #torch.Size([B, max_length, 30522])

class PetCriterion(nn.Module):
    def __init__(self, config, m2c_tensor, filler_len):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.m2c = m2c_tensor.to(config.device) #[label_size, max_num_verbalizers]
        self.filler_len = filler_len.to(config.device) #[label_size]

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:

        """ for example
        VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech", "Science", "Scientific"]
    }
        m2c [label_size, max_num_verbalizers] 一共四个类别，2088代表
        m2c: [[2088, -1, -1],
              [2998, -1, -1],
              [2499, -1, -1],
              [6627, 2671, 4045]]
        # 2088代表world在vocab中的索引, 2998代表Sports在vocab中的索引, 2499代表Business在vocab中的索引
        # 6627, 2671, 4045分别表示"Tech", "Science", "Scientific" 在vocab中的索引
        torch.zeros_like(self.m2c) = [[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]
        torch.max(torch.zeros_like(self.m2c), self.m2c) = [[2088, 0, 0],
                                                           [2998, 0, 0],
                                                           [2499, 0, 0],
                                                           [6627, 2671, 4045]]
       cls_logits =  logits[torch.max(torch.zeros_like(self.m2c), self.m2c)]= [[ 1.9206, -5.6852, -5.6852],
                                                                               [ 0.5106, -5.6852, -5.6852],
                                                                               [-0.4168, -5.6852, -5.6852],
                                                                               [-4.4497, -0.3091, -3.1954]]

        (self.m2c > 0) = [[ True, False, False],
                          [ True, False, False],
                          [ True, False, False],
                          [ True,  True,  True]]
       cls_logits * (self.m2c > 0).float() = [[ 1.9206, -0.0000, -0.0000],
                                              [ 0.5106, -0.0000, -0.0000],
                                              [-0.4168, -0.0000, -0.0000],
                                              [-4.4497, -0.3091, -3.1954]]

       filler_len [1., 1., 1., 3.] label为0，1,2都只有一个verbalizers与之对应，但是label为3的有三个verbalizers与之对应
        """


        # m2c [label_size, max_num_verbalizers]
        # torch.max(torch.zeros_like(self.m2c), self.m2c) [label_size, max_num_verbalizers] 过滤掉m2c中值为-1的
        # cls_logits [label_size, max_num_verbalizers] 每个label对应的logits
        cls_logits = logits[torch.max(torch.zeros_like(self.m2c), self.m2c)]

        # self.m2c > 0 [label_size, max_num_verbalizers] 返回bool值，m2c中大于0的返回True， 小于等于0的返回False
        # cls_logits * (self.m2c > 0)  [label_size, max_num_verbalizers] 过滤等于掉0
        cls_logits = cls_logits * (self.m2c > 0).float()

        cls_logits = cls_logits.sum(axis=1) / self.filler_len #verbalizers多的加和求平均
        return cls_logits#[label_size] 表示每个当前logits对应每个类别的值

    def predict(self, cls_logits):
        cls_logits = cls_logits.cpu()
        cls_logits = cls_logits.detach()
        cls_logits = cls_logits.numpy()
        predictions = np.argmax(cls_logits, axis=1)

        return predictions


    def forward(self, logits, mlm_labels, labels):
        """
        logits: [B, max_length, vocab_size]
        mlm_labels: [B, max_length] 元素为1或者-1

        """
        masked_logits = logits[mlm_labels >= 0] # [B, vocab_size]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in
                                  masked_logits]) #[B, label_size]
        predictions = self.predict(cls_logits)
        loss = self.criterion(cls_logits, labels.view(-1))
        return loss, predictions





