#!/usr/bin/env bash
# -*- coding: utf-8 -*-

repo_path=.
bert_path=$repo_path/FPT
data_dir=$repo_path/data
export PYTHONPATH="$PYTHONPATH:$repo_path"

ckpt_path=Your/Path/.Ckpt
val_sighan=15

output_dir=outputs_copy/predict
mkdir -p $output_dir

model_type=gate_copy_retr
name_retrieve=knn

dstore_dir=$repo_path/dstore_${val_sighan}
dstore_maxsize=23591250
lamb=0.3
temperature=10

CUDA_VISIBLE_DEVICES=0  python -u finetune/predict.py \
  --bert_path $bert_path \
  --ckpt_path $ckpt_path \
  --data_dir $data_dir \
  --label_file $data_dir/test.sighan${val_sighan}.lbl.tsv \
  --save_path $output_dir \
  --gpus=0, \
  --model_type $model_type \
  --name_retrieve $name_retrieve \
  --dstore_dir $dstore_dir \
  --dstore_maxsize $dstore_maxsize \
  --lamb $lamb \
  --temperature $temperature 
  
