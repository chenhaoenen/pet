#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ..
pwdPath="$(pwd)"

MODEL_PATH=/workspace/pretrained-model/bert-base-cased
TRAIN_DATA_PATH=/workspace/mengtaosun/liheng/pet/data/positive_sample1.csv
EVAL_DATA_PATH=/workspace/mengtaosun/liheng/pet/data/positive_sample_with_label.csv

python -m src.train \
  --epochs 50 \
  --batch_size 8 \
  --max_seq_length 256 \
  --learning_rate 5e-5 \
  --log_freq 100 \
  --eval_freq 500 \
  --model_path $MODEL_PATH \
  --train_data_path  $TRAIN_DATA_PATH \
  --eval_data_path $EVAL_DATA_PATH
