#!/usr/bin/env bash
# -*- coding: utf-8 -*-

repo_path=.
bert_path=$repo_path/FPT
data_dir=$repo_path/data
export PYTHONPATH="$PYTHONPATH:$repo_path"

ckpt_path=Your/Path/.Ckpt
val_sighan=15

output_dir=$repo_path/dstore_${val_sighan}
mkdir -p $output_dir

name_retrieve_prepare=knn
name_train=train_all
text_type=both

CUDA_VISIBLE_DEVICES=0  python -u retrieve/prepare_dstore.py \
  --bert_path $bert_path \
  --ckpt_path $ckpt_path \
  --data_dir $data_dir \
  --save_path $output_dir \
  --name_retrieve_prepare $name_retrieve_prepare \
  --name_train $name_train \
  --text_type $text_type \
  --gpus=0,

CUDA_VISIBLE_DEVICES=0  python -u retrieve/train_dstore.py \
  --save_path $output_dir \
  --name_retrieve_prepare $name_retrieve_prepare \
  --text_type $text_type \
  --gpus=0,
