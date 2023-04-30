# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/4/29 22:16
# Description:  
# --------------------------------------------
import torch
import torch.nn as nn
from transformers.modeling_bert import BertForSequenceClassification

from src.dataset import AgNewsDataset

class PetModel(nn.Module):
    def __int__(self, config):
        self.model = BertForSequenceClassification.from_pretrained(config.model_path)

    def forword(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return out #torch.Size([B, max_length, 30522])

class PetCriterion(nn.Module):
    def __int__(self, config, m2c_tensor, filler_len):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.label_list = AgNewsDataset.VERBALIZER
        self.m2c = m2c_tensor #[label_size, max_num_verbalizers]
        self.filler_len = filler_len #[label_size]

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:

        cls_logits = logits[torch.max(torch.zeros_like(self.m2c), self.m2c)] # m2c中有一些是被填充的-1， 其他的是有真实值的 #torch.max(torch.zeros_like(m2c), m2c)可以将 大于等于0的索引拿到
        cls_logits = cls_logits * (m2c > 0).float() #这里(m2c > 0) 又一次过滤掉了那些等于0的，因此最终只留下那些索引大于0的label真正对应的logits值

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len # 如果某个label的verbalize比较多，则将对应的logits加起来作为最终label的 logits
        return cls_logits

    def forword(self, logits, mlm_labels):
        """
        logits: model output #torch.Size([B, max_length, vocab_size])
        mlm_labels: # mlm_labels torch.Size([B, max_length)
        """
        masked_logits = logits[mlm_labels >= 0] # torch.Size([B, vocab_size])
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in
                                  masked_logits])  # 这里不batch的处理，因为每个样本对应的 verbalize个数可能不同
        pass

