import csv

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

VERBALIZER = {
    ('0', '0', '0', '0', '0', '1'): ['jews', 'people', 'like', 'gay', 'wikipedia', 'jewish', 'islam', 'page', 'know',
                                     'muslims'],
    ('0', '0', '0', '0', '1', '0'): ['daedalus', 'biznitch', 'repeat', 'like', 'wikipedia', 'people', 'page', 'know',
                                     'think', 'time'],
    ('0', '0', '0', '0', '1', '1'): ['wait', 'think', 'wikipedia', 'like', 'pakis', 'page', 'funny', 'sorbs', 'little',
                                     'people'],
    ('0', '0', '0', '1', '0', '0'): ['know', 'll', 'wikipedia', 'going', 'offended', 'good', 've', 'life', 'vandalize',
                                     'person'],
    ('0', '0', '1', '0', '0', '0'): ['page', 'like', 'wikipedia', 'article', 'talk', 'user', 'people', 'think', 'edit',
                                     'know'],
    ('0', '0', '1', '0', '1', '0'): ['twat', 'neiln', 'bbb', 'dicks', 'licks', 'like', 'wikipedia', 'article', 'user',
                                     'page'],
    ('0', '0', '1', '0', '1', '1'): ['user', 'dont', 'way', 'stop', 'truth', 'people', 'queen', 'monkey', 'nigs',
                                     'wikipedia'],
    ('1', '0', '0', '0', '0', '0'): ['hate', 'wikipedia', 'like', 'aids', 'gay', 'page', 'people', 'dont', 'care',
                                     'know'],
    ('1', '0', '0', '0', '0', '1'): ['gay', 'utc', 'cody', 'like', 'people', 'want', 'wikipedia', 'jews', 'stop',
                                     'hate'],
    ('1', '0', '0', '0', '1', '0'): ['moron', 'hi', 'hate', 'admin', 'bad', 'like', 'old', 'wikipedia', 'cougar',
                                     'know'],
    ('1', '0', '0', '0', '1', '1'): ['jew', 'fat', 'nigger', 'tommy', 'gay', 'like', 'good', 'money', 'making',
                                     'bunch'],
    ('1', '0', '0', '1', '0', '0'): ['kill', 'die', 'going', 'll', 'death', 'edit', 'shoot', 'stop', 'dead', 'hope'],
    ('1', '0', '0', '1', '1', '0'): ['die', 'll', 'going', 'hope', 'supertr', 'kill', 'pathetic', 'send', 'forever',
                                     'respect'],
    ('1', '0', '1', '0', '0', '0'): ['bullshit', 'nipple', 'fuck', 'penis', 'buttsecks', 'shit', 'poop', 'boobs',
                                     'wikipedia', 'fucking'],
    ('1', '0', '1', '0', '0', '1'): ['nigger', 'white', 'jewranger', 'like', 'gay', 'page', 'black', 'boy', 'racist',
                                     'im'],
    ('1', '0', '1', '0', '1', '0'): ['fuck', 'fucking', 'faggots', 'freedom', 'bastered', 'know', 'like', 'shit',
                                     'wikipedia', 'page'],
    ('1', '0', '1', '0', '1', '1'): ['nigger', 'jew', 'fat', 'cunt', 'fucking', 'like', 'fuck', 'gay', 'spanish',
                                     'licker'],
    ('1', '0', '1', '1', '0', '0'): ['ŷour', 'ŵill', 'kill', 'ŷou', 'ţhe', 'iţ', 'time', 'll', 'wish', 'unblock'],
    ('1', '0', '1', '1', '1', '0'): ['ban', 'lifetime', 'fuck', 'll', 'fucking', 'die', 'supertr', 'kill', 'fuckin',
                                     'live'],
    ('1', '0', '1', '1', '1', '1'): ['die', 'ass', 'fucking', 'kill', 'fuck', 'im', 'shit', 'gay', 'bitch', 'stupid'],
    ('1', '1', '0', '0', '0', '0'): ['sucks', 'criminalwar', 'bush', 'anthony', 'bradbury', 'like', 'wikipedia',
                                     'communism', 'shit', 'penis'],
    ('1', '1', '0', '0', '1', '0'): ['faggot', 'moron', 'hi', 'god', 'chocobos', 'cheese', 'hell', 'bush', 'want',
                                     'george'],
    ('1', '1', '0', '1', '0', '0'): ['kill', 'jim', 'wales', 'die', 'talk', 'blank', 'page', 'rvv', 'continue',
                                     'block'],
    ('1', '1', '1', '0', '0', '0'): ['fuck', 'shit', 'fucksex', 'ass', 'offfuck', 'wikipedia', 'anal', 'rape',
                                     'chester', 'marcolfuck'],
    ('1', '1', '1', '0', '1', '0'): ['fuck', 'suck', 'fucking', 'bitch', 'cunt', 'ass', 'yourselfgo', 'fucker', 'sucks',
                                     'shit'],
    ('1', '1', '1', '0', '1', '1'): ['nigger', 'fuck', 'faggot', 'huge', 'suck', 'mexicans', 'niggas', 'stupid', 'ass',
                                     'shit'],
    ('1', '1', '1', '1', '1', '0'): ['ass', 'die', 'block', 'wikipedia', 'kill', 'page', 'talk', 'filter', 'dust',
                                     'rvv'],
    ('1', '1', '1', '1', '1', '1'): ['die', 'di', 'edie', 'wikipedia', 'wiki', 'en', 'org', 'fuck', 'bitch', 'ass']
}
VERBALIZER_INDEX_LABEL = {
    ('0', '0', '0', '0', '0', '1'): 0,
    ('0', '0', '0', '0', '1', '0'): 1,
    ('0', '0', '0', '0', '1', '1'): 2,
    ('0', '0', '0', '1', '0', '0'): 3,
    ('0', '0', '1', '0', '0', '0'): 4,
    ('0', '0', '1', '0', '1', '0'): 5,
    ('0', '0', '1', '0', '1', '1'): 6,
    ('1', '0', '0', '0', '0', '0'): 7,
    ('1', '0', '0', '0', '0', '1'): 8,
    ('1', '0', '0', '0', '1', '0'): 9,
    ('1', '0', '0', '0', '1', '1'): 10,
    ('1', '0', '0', '1', '0', '0'): 11,
    ('1', '0', '0', '1', '1', '0'): 12,
    ('1', '0', '1', '0', '0', '0'): 13,
    ('1', '0', '1', '0', '0', '1'): 14,
    ('1', '0', '1', '0', '1', '0'): 15,
    ('1', '0', '1', '0', '1', '1'): 16,
    ('1', '0', '1', '1', '0', '0'): 17,
    ('1', '0', '1', '1', '1', '0'): 18,
    ('1', '0', '1', '1', '1', '1'): 19,
    ('1', '1', '0', '0', '0', '0'): 20,
    ('1', '1', '0', '0', '1', '0'): 21,
    ('1', '1', '0', '1', '0', '0'): 22,
    ('1', '1', '1', '0', '0', '0'): 23,
    ('1', '1', '1', '0', '1', '0'): 24,
    ('1', '1', '1', '0', '1', '1'): 25,
    ('1', '1', '1', '1', '1', '0'): 26,
    ('1', '1', '1', '1', '1', '1'): 27,
}
VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

class AgNewsDataset(Dataset):
    def __init__(self, data_path, model_path, pattern_id, max_length):
        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # 加载tokenizer
        self.mask = self.tokenizer.mask_token  # 获取tokenizer中mask标签</mask>
        self.mask_id = self.tokenizer.mask_token_id  # 词表中</mask>对应的id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue

                if len(row) == 7:
                    comment_text, label0, label1, label2, label3, label4, label5 = row
                else:
                    comment_text, label0, label1, label2, label3, label4, label5, _ = row

                if (label0, label1, label2, label3, label4, label5) not in VERBALIZER_INDEX_LABEL: continue
                example = [comment_text, VERBALIZER_INDEX_LABEL[(label0, label1, label2, label3, label4, label5)]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    # 得到每个VERBALIZER的ids
    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 #[len(VERBALIZER_LABEL), max_num_verbalizers]所有值都为-1
        for label_idx, verbalizers in VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor #[label_size, max_num_verbalizers]
    # 返回每个label对应的词表长度
    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len #[label_size] 有的label 的verbalize比较多，有的label的verbalize较少

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids

    # 把文本和模板结合起来并转化成ids
    def encode(self, comment_text):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', comment_text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'News:', comment_text]
        elif self.pattern_id == 2:
            prompt_text = [comment_text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', comment_text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', comment_text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', comment_text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    # 得到被mask掉的位置
    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    # 调用前面的函数得到BERT的参数
    def __getitem__(self, idx):
        comment_text, label = self.examples[idx]
        feature = self.encode(comment_text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, mlm_labels, labels = zip(*batch)
    input_ids = torch.stack([w.squeeze() for w in input_ids])
    token_type_ids = torch.stack([w.squeeze() for w in token_type_ids])
    attention_mask = torch.stack([w.squeeze() for w in attention_mask])
    mlm_labels = torch.stack([torch.Tensor(mlm_label).long() for mlm_label in mlm_labels])
    labels = torch.stack([torch.Tensor([label]).long() for label in labels])

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels
