#!/bin/bash



python3 cli.py \
 --method pet \
 --pattern_ids 0 1 2 3 4 \
 --data_dir data/ag_news_csv \
 --model_type bert \
 --model_name_or_path model/bert-base-uncased \
 --task_name agnews \
 --output_dir output/agnews \
 --do_train