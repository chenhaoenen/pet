#Pet
## Introduction
> 该实例仅仅是pet mlm模型的抽取，具体的实现，需要配合自己的需求来增加相应的代码

### 目录结构
```text
├─data
│  ├─ag_news_csv
│  │  ├─readme.txt
│  │  ├─test.csv
│  │  ├─train.csv
│  │  └─classes.txt
│  ├─ag_news_csv.tar.gz
│  └─ag_news_csv.tar
├─src
│  ├─utils.py
│  ├─train.py
│  ├─model.py
│  └─dataset.py
├─output
│  └─agnews
│     └─p0-i0
├─model
│  └─bert-base-uncased #需要根据自己需求下载
│     ├─tokenizer_config.json
│     ├─tokenizer.json
│     ├─gitattributes
│     ├─config.json
│     ├─vocab.txt
│     └─pytorch_model.bin
├─.gitignore
├─README.md
└─requirements.txt
```


## Install
> pip install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/
> 模型需要自行下载，放在model/文件夹下
> 数据也需要自行下载，放在data/文件夹下


## Running

### 脚本执行
```shell
cd src
python3 -m src.train
```
### pycharm执行
直接执行train.py即可
