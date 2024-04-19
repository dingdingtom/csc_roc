import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import Counter
from pytorch_lightning import Trainer
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from finetune.train import CSCTask
from datasets.bert_csc_dataset import RetrieveDataset
from datasets.collate_functions import collate_to_max_length_for_train_dynamic_pron_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", default=os.path.join('..', 'FPT'), type=str, help="bert config file")
    parser.add_argument("--data_dir", default=os.path.join('..', 'data'), type=str, help="train data path")
    parser.add_argument("--ckpt_path", default=None, type=str, help="resume_from_checkpoint")
    parser.add_argument("--save_path", default=os.path.join('..', 'dstore'), type=str)
    parser.add_argument("--name_record_maxsize", default='latest_maxsize.txt', type=str)
    parser.add_argument("--name_train", default='train_all_people', type=str)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--name_retrieve_prepare", default='none', type=str)
    parser.add_argument("--num_repeat", default=-1, type=int)
    parser.add_argument("--text_type", default='both', type=str)

    parser.add_argument("--model_type", default='lm', type=str)
    parser.add_argument("--name_retrieve", default='none', type=str)
    parser.add_argument("--dim_hidden", default=768, type=int)
    parser.add_argument("--num_bit", default=16, type=int)
    parser.add_argument("--workers", type=int, default=8, help="num workers for dataloader")

    return parser

def isValid(char):
    lst_char_ignore = [
        '，', '。', '？', '！', '、', '~', '—', '（', '）', '【', '】', 
        ',', '.', '?', '!', '-', '(', ')', '[', ']', '{', '}', '#', '+', '/', '*', '=',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '%', '$', '￥', '@', '#',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    ]
    # for char_ignore in lst_char_ignore:
    #     if char_ignore in char:
    #         return False 
    return True

if __name__ == '__main__':
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    device = torch.device('cuda:0')

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # model
    model = CSCTask(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    # data
    path_train = os.path.join(args.data_dir, args.name_train)
    df = pd.read_json(path_train, lines=True, )  #nrows=50
    dataset = RetrieveDataset(df)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_to_max_length_for_train_dynamic_pron_loss, fill_values=[0, 0, 0, 0, 0]),
    )

    pbar = tqdm(total=len(dataloader))
    dic_char_pinyin_to_cnt = {}
    dstore_maxsize = 0
    for i_batch, batch in enumerate(dataloader):
        input_ids, pinyin_ids, label, tgt_pinyin_ids, pinyin_label = batch 
        attention_mask = ((input_ids != 0) * (input_ids != 100) * (input_ids != 101) * (input_ids != 102)).long()
        B, T = input_ids.shape
        # input_ids, label, attention_mask: (B, T)
        # pinyin_ids, tgt_pinyin_ids: (B, 8 * T)
        # pinyin_label: (B, T, 3)
        pinyin_ids, tgt_pinyin_ids = pinyin_ids.reshape(B, T, 8), tgt_pinyin_ids.reshape(B, T, 8)
        
        if args.num_repeat == -1:
            N = attention_mask.sum()
            dstore_maxsize += N
        else:
            decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decoded_tgt = tokenizer.batch_decode(label, skip_special_tokens=False)
            src = [''.join(decoded_one.split(' ')) for decoded_one in decoded]
            tgt = [''.join(decoded_one.split(' ')) for decoded_one in decoded_tgt]
            lst_chars = [tokenizer.tokenize(src_one) for src_one in src]
            lst_chars_tgt = [tokenizer.tokenize(tgt_one) for tgt_one in tgt]

            for i_row in range(B):
                for i_char in range(T):
                    if attention_mask[i_row, i_char] == 0:
                        continue 
                    char = lst_chars[i_row][i_char]
                    char_tgt = lst_chars_tgt[i_row][i_char]
                    if not isValid(char_tgt):
                        continue
                    input_ids_char = input_ids[i_row, i_char]
                    pinyin_ids_char = pinyin_ids[i_row, i_char]
                    label_char = label[i_row, i_char]
                    tgt_pinyin_ids_char = tgt_pinyin_ids[i_row, i_char]
                    pinyin_label_char = pinyin_label[i_row, i_char]
                    key = (char_tgt, label_char.item(), tuple(tgt_pinyin_ids_char.tolist()))
                    if args.num_repeat != -1 and dic_char_pinyin_to_cnt.get(key, 0) >= args.num_repeat:
                        continue
                    dic_char_pinyin_to_cnt[key] = dic_char_pinyin_to_cnt.get(key, 0) + 1
                    dstore_maxsize += 1
        pbar.update(1)
    pbar.close()
    if args.text_type == 'both':
        dstore_maxsize *= 2

    # dstore
    path_dstore_keys = os.path.join(args.save_path, f'{args.name_retrieve_prepare}_{args.text_type}_keys_b{args.num_bit}_ds{dstore_maxsize}.npy')
    path_dstore_values = os.path.join(args.save_path, f'{args.name_retrieve_prepare}_{args.text_type}_values_b{args.num_bit}_ds{dstore_maxsize}.npy')

    dtype_float = np.float32 if args.num_bit == 32 else np.float16
    dtype_long = np.int64
    dstore_keys = np.memmap(path_dstore_keys, dtype=dtype_float, mode='w+', shape=(dstore_maxsize, args.dim_hidden))
    dstore_values = np.memmap(path_dstore_values, dtype=dtype_long, mode='w+', shape=(dstore_maxsize))
    
    pbar = tqdm(total=len(dataloader))
    dic_char_pinyin_to_cnt = {}
    dstore_cursize = 0
    for i_batch, batch in enumerate(dataloader):
        input_ids, pinyin_ids, label, tgt_pinyin_ids, pinyin_label = batch 
        attention_mask = ((input_ids != 0) * (input_ids != 100) * (input_ids != 101) * (input_ids != 102)).long()
        B, T = input_ids.shape
        # input_ids, label, attention_mask: (B, T)
        # pinyin_ids, tgt_pinyin_ids: (B, 8 * T)
        # pinyin_label: (B, T, 3)
        pinyin_ids, tgt_pinyin_ids = pinyin_ids.reshape(B, T, 8), tgt_pinyin_ids.reshape(B, T, 8)
        
        with torch.no_grad():
            outputs_x = model.model.bert(
                input_ids.to(device),
                pinyin_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            encoded_x = outputs_x[0]  # (B, T, H)
            outputs_y = model.model.bert(
                label.to(device),
                tgt_pinyin_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            encoded_y = outputs_y[0]  # (B, T, H)

        if args.num_repeat == -1:
            decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decoded_tgt = tokenizer.batch_decode(label, skip_special_tokens=False)
            src = [''.join(decoded_one.split(' ')) for decoded_one in decoded]
            tgt = [''.join(decoded_one.split(' ')) for decoded_one in decoded_tgt]
            lst_chars = [tokenizer.tokenize(src_one) for src_one in src]
            lst_chars_tgt = [tokenizer.tokenize(tgt_one) for tgt_one in tgt]

            for i_row in range(B):
                for i_char in range(T):
                    if attention_mask[i_row, i_char] == 0:
                        continue 
                    char = lst_chars[i_row][i_char]
                    char_tgt = lst_chars_tgt[i_row][i_char]
                    if not isValid(char_tgt):
                        attention_mask[i_row, i_char] = 0
                        continue
                    input_ids_char = input_ids[i_row, i_char]
                    pinyin_ids_char = pinyin_ids[i_row, i_char]
                    label_char = label[i_row, i_char]
                    tgt_pinyin_ids_char = tgt_pinyin_ids[i_row, i_char]
                    pinyin_label_char = pinyin_label[i_row, i_char]
                    key = (char_tgt, label_char.item(), tuple(tgt_pinyin_ids_char.tolist()))
                    if args.num_repeat != -1 and dic_char_pinyin_to_cnt.get(key, 0) >= args.num_repeat:
                        attention_mask[i_row, i_char] = 0
                        continue
                    dic_char_pinyin_to_cnt[key] = dic_char_pinyin_to_cnt.get(key, 0) + 1
            N = int(attention_mask.sum())

        if args.text_type == 'both':
            lst_encoded = [encoded_x, encoded_y]
        elif args.text_type == 'src':
            lst_encoded = [encoded_x]
        elif args.text_type == 'tgt':
            lst_encoded = [encoded_y]
        
        for encoded in lst_encoded:
            keys = encoded[attention_mask == 1]  # (N, H)
            values = label[attention_mask == 1]  # (N,)
            if dstore_cursize + N > dstore_maxsize:
                N = dstore_maxsize - dstore_cursize
                keys, values = keys[:N], values[:N]
            f_float = lambda x: x.cpu().numpy().astype(dtype_float)
            f_long = lambda x: x.cpu().numpy().astype(dtype_long)
            keys, values = f_float(keys), f_long(values)
            dstore_keys[dstore_cursize:dstore_cursize + N] = keys
            dstore_values[dstore_cursize:dstore_cursize + N] = values
            dstore_cursize += N
            pbar.set_description(f'{dstore_cursize}/{dstore_maxsize}')

        pbar.update(1)
        if dstore_cursize == dstore_maxsize:
            break
    pbar.close()

    path_record_maxsize = os.path.join(args.save_path, args.name_record_maxsize)
    with open(path_record_maxsize, 'w') as o:
        o.write(f'{dstore_maxsize}')
