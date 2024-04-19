#!/usr/bin/env bash
# -*- coding: utf-8 -*-

repo_path=.
bert_path=$repo_path/FPT
data_dir=$repo_path/data
export PYTHONPATH="$PYTHONPATH:$repo_path"

ckpt_path=Your/Path/.Ckpt
val_sighan=15

lr=5e-5
bs=16
accumulate_grad_batches=4
epoch=30
output_dir=$repo_path/outputs
model_type=gate_copy_retr
name_retrieve=knn
dstore_dir=$repo_path/dstore_${val_sighan}
dstore_maxsize=23591250
lamb=0.3
temperature=10

mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0 python -u $repo_path/finetune/train.py \
--bert_path $bert_path \
--ckpt_path $ckpt_path \
--data_dir $data_dir \
--label_file $data_dir/test.sighan${val_sighan}.lbl.tsv \
--save_path $output_dir \
--max_epoch $epoch \
--lr $lr \
--warmup_proporation 0.1 \
--batch_size $bs \
--accumulate_grad_batches $accumulate_grad_batches  \
--model_type $model_type \
--name_retrieve $name_retrieve \
--dstore_dir $dstore_dir \
--dstore_maxsize $dstore_maxsize \
--gpus=0, \
--reload_dataloaders_every_n_epochs 1 
