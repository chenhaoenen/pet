# PET

## 目录结构
```angular2html
├── README.md
├── ckpts（checkpoint 目录）
│   ├── 2023-07-18T09-20-26
│   │   └── epoch1_step1000_acc0.310013.pt
├── data （数据目录）
│   ├── positive_sample1.csv
│   └── positive_sample_with_label.csv
├── log （日志目录）
│   └── 2023-07-18_09-37-31_068990.log
├── model （pretrained model 目录）
├── requirements.txt 
├── scripts （运行脚本目录）
│   └── submit.sh
│   └── submit_lr1.5e-5.sh
├── src （代码目录）
│   ├── dataset.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── tensorboard （tensorboard 目录）
    ├── 2023-07-18T09-15-27
    │   └── events.out.tfevents.1689671727.e2c5d025e223.1984.0
```

## Install
> 主要的安装包在 `requirements.txt`  中

```shell
pip install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/
```

## Running
```shell
cd scripts

bash submit.sh
```

> 如果要修改相应参数最好不要源码，
> 可以在 `scripts/submit.sh` 中修改相应参数并执行，
> 或者复制 `submit.sh` 修改相应的参数并执行(例如修改learning rate 为1.5e-5)将文件命名为submit_lr1.5e-5.sh 然后执行
```shell
cd scripts

bash submit_lr1.5e-5.sh
```

