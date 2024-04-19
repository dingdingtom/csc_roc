#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
from functools import partial
from attr import has
from pypinyin import pinyin
from copy import deepcopy
from multiprocessing import freeze_support

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from metrics.metric import Metric
from utils.random_seed import set_random_seed
from retrieve.retrievers import KnnRetriever, KnnRetrieverArgs
from models.modeling_multitask import Dynamic_GlyceBertForMultiTask
from datasets.bert_csc_dataset import TestCSCDataset, Dynaimic_CSCDataset
from datasets.collate_functions import collate_to_max_length_with_id, collate_to_max_length_for_train_dynamic_pron_loss

set_random_seed(2333)


def decode_sentence_and_get_pinyinids(ids):
    dataset = TestCSCDataset(
        data_path='data/test.sighan15.pkl',
        chinese_bert_path='FPT',
    )
    sent = ''.join(dataset.tokenizer.decode(ids).split(' '))
    tokenizer_output = dataset.tokenizer.encode(sent)
    pinyin_tokens = dataset.convert_sentence_to_pinyin_ids(sent, tokenizer_output)
    pinyin_ids = torch.LongTensor(pinyin_tokens).unsqueeze(0)
    return sent, pinyin_ids


class CSCTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False)
        self.model = Dynamic_GlyceBertForMultiTask.from_pretrained(self.bert_dir)
        self.model.model_type = args.model_type
        if args.ckpt_path is not None:
            print("loading from ", args.ckpt_path)
            ckpt = torch.load(args.ckpt_path,)["state_dict"]
            new_ckpt = {}
            for key in ckpt.keys():
                new_ckpt[key[6:]] = ckpt[key]
            self.model.load_state_dict(new_ckpt,strict=False)
            print(self.model.device, torch.cuda.is_available())
        self.vocab_size = self.bert_config.vocab_size
        self.model.bert2 = deepcopy(self.model.bert)
        self.model.cls2 = deepcopy(self.model.cls)

        self.loss_fct = CrossEntropyLoss()
        self.num_gpus = len(self.args.gpus.strip(',').split(','))
        self.retriever = self.getRetriever(args)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, labels=None, pinyin_labels=None, tgt_pinyin_ids=None, var=1):
        attention_mask = (input_ids != 0).long()
        output_hidden_states = self.args.name_retrieve != 'none'
        return self.model(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            labels=labels,
            tgt_pinyin_ids=tgt_pinyin_ids, 
            pinyin_labels=pinyin_labels,
            gamma=self.args.gamma if 'gamma' in self.args else 0,
            output_hidden_states=output_hidden_states,
            retriever=self.retriever,
        )
    
    def getRetriever(self, args):
        if args.name_retrieve == 'none':
            retriever = None
        elif args.name_retrieve == 'knn':
            args_retriever = KnnRetrieverArgs(
                dstore_dir=args.dstore_dir,
                dstore_maxsize=args.dstore_maxsize,
                lamb=args.lamb,
                ngpu=args.ngpu,
                temperature=args.temperature,
                num_centroid=args.num_centroid,
            )
            retriever = KnnRetriever(args_retriever)
        return retriever
    
    def getPredictLabels(self, input_ids, pinyin_ids):
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        B, T = input_ids.shape
        pinyin_ids = pinyin_ids.view(B, T, 8)

        if self.args.model_type in ['lm', 'phon']:
            with torch.no_grad():
                attention_mask = (input_ids != 0).long()
                outputs_x = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    output_hidden_states=True,
                )
                encoded_x = outputs_x[0]  # (B, T, H)
                prediction_scores = self.model.cls(encoded_x)[0]  # (B, T, V)
                logits = prediction_scores
            predict_scores = F.softmax(logits, dim=-1)
            predict_labels = torch.argmax(predict_scores, dim=-1) * mask
            return predict_labels
        
        elif self.args.model_type == 'gate_copy_retr':
            with torch.no_grad():
                attention_mask = (input_ids != 0).long()
                outputs_x = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    output_hidden_states=True,
                )
                outputs_x2 = self.model.bert2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    output_hidden_states=True,
                )

                encoded_x = outputs_x[0]  # (B, T, H)
                encoded_x2 = outputs_x2[0]  # (B, T, H)
                prediction_scores = self.model.cls2(encoded_x2)[0]  # (B, T, V)
            V = prediction_scores.shape[2]
            device = encoded_x.device
        
            with torch.no_grad():
                gate_inputs = encoded_x2
                gate_scores = self.model.net_gate(encoded_x2)  # (B, T, 2)
                copy_gate_scores = self.model.net_copy_gate(gate_inputs)  # (B, T, 2)
            gate_probs = torch.softmax(gate_scores, dim=2)  # (B, T, 2)
            masks_retrieve = gate_probs[:, :, 0] < 0.5  # (B, T)
            masks_copy = ~masks_retrieve  # (B, T)
            masks_copy = (masks_copy.int() * mask).bool()
            masks_retrieve = (masks_retrieve.int() * mask).bool()
            masks_both = torch.bitwise_or(masks_copy, masks_retrieve)
            num_copy, num_retrieve = masks_copy.sum(), masks_retrieve.sum()
            seq_logprobs = F.log_softmax(prediction_scores, dim=2)  # (B, T, V)
            logits_new = seq_logprobs

            copy_probs = torch.zeros(B, T, V).to(device)  # (B, T, V)
            copy_probs = torch.scatter_add(copy_probs, dim=2, index=input_ids.unsqueeze(2), src=torch.ones(B, T, 1).to(device))
            copy_logprobs = torch.log(torch.clamp(copy_probs, min=1e-30))[masks_copy]  # (N_copy, V)

            if num_copy:
                copy_gate_logprobs = F.log_softmax(copy_gate_scores, dim=-1)[masks_copy].reshape(num_copy, 2, 1)  # (N_copy, 2, 1)
                copy_all_logprobs = torch.stack([copy_logprobs, seq_logprobs[masks_copy]], dim=1)   # (N_copy, 2, V)
                copy_edit_logprobs = torch.logsumexp(copy_gate_logprobs + copy_all_logprobs, dim=1)  # (N_copy, V)
                logits_new[masks_copy] = copy_edit_logprobs

            retriever = self.retriever
            if num_retrieve:
                retrieve_input = encoded_x
                retrieve_input = retrieve_input[masks_retrieve].unsqueeze(1)  # (N_retr, 1, H)
                retrieve_input = F.normalize(retrieve_input, p=2, dim=2)
                dic_retrieve = retriever.retrieve(retrieve_input)
                retrieve_outputs = retriever.getRetrieveOutputs(dic_retrieve)
                
                retrieve_embeds, retrieve_values = (
                    retrieve_outputs['retrieve_embeds'].squeeze(1).float().to(device),  # (N_retr, K, H)
                    retrieve_outputs['retrieve_values'].squeeze(1).to(device),  # (N_retr, K)
                )
                K = retrieve_values.shape[1]


                cross_scores = torch.bmm(retrieve_input, retrieve_embeds.transpose(1, 2)).reshape(num_retrieve, K)  # (num_mask, num_k)
                cross_probs = torch.softmax(cross_scores, dim=1)  # (num_mask, num_k)
                
                retrieve_probs = torch.zeros(num_retrieve, K, V).to(device)  # (num_mask, num_k, vocab_size)
                retrieve_probs = torch.scatter(retrieve_probs, dim=2, index=retrieve_values.unsqueeze(2), src=torch.ones(num_retrieve, K, 1).float().to(device))
                retrieve_probs_agg = (cross_probs.unsqueeze(2) * retrieve_probs).sum(dim=1)  # (num_mask, vocab_size)
                retrieve_logprobs_agg = torch.log(torch.clamp(retrieve_probs_agg, min=1e-30))  # (num_mask, vocab_size)

                retrieve_gate_scores = self.model.net_retrieve_gate(gate_inputs)  # (B, T, 2)
                retrieve_gate_probs = torch.softmax(retrieve_gate_scores, dim=-1)[masks_retrieve].reshape(num_retrieve, 2)
                retrieve_embeds_agg = (cross_probs.unsqueeze(2) * retrieve_embeds).sum(dim=1).to(device)  # (num_mask, dim_hidden)
                encoded_lm = encoded_x2[masks_retrieve].reshape(num_retrieve, -1)
                encoded_all = torch.stack([encoded_lm, retrieve_embeds_agg], dim=1)  # (num_mask, 2, dim_hidden)
                encoded_agg = (retrieve_gate_probs.unsqueeze(2) * encoded_all).sum(dim=1)
                retrieve_prediction_scores = self.model.cls2(encoded_agg)[0]
                retrieve_edit_logprobs = F.log_softmax(retrieve_prediction_scores, dim=1)
                logits_new[masks_retrieve] = retrieve_edit_logprobs

            predict_scores = F.softmax(logits_new, dim=-1)
            predict_labels = torch.argmax(predict_scores, dim=-1) * mask
            return predict_labels

    def compute_loss(self, batch):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        B, T = input_ids.shape
        pinyin_ids = pinyin_ids.view(B, T, 8)
        tgt_pinyin_ids = tgt_pinyin_ids.view(B, T, 8)
        outputs = self.forward(
            input_ids, pinyin_ids, labels=labels, pinyin_labels=pinyin_labels, tgt_pinyin_ids=tgt_pinyin_ids, 
            var= self.args.var if 'var' in self.args else 1
        )
        loss = outputs.loss
        logits = outputs.logits
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        tf_board_logs = {
            "train_loss": loss.item(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tgts, tokens_size = batch
        predict_labels = self.getPredictLabels(input_ids, pinyin_ids)
        if '13' in self.args.label_file:
            predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        return {
            "tgt_idx": labels.cpu(),
            "pred_idx": predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }

    def validation_epoch_end(self, outputs):
        metric = Metric(vocab_path=self.args.bert_path)
        pred_txt_path = os.path.join(self.args.save_path, "preds.txt")
        pred_lbl_path = os.path.join(self.args.save_path, "labels.txt")
        if len(outputs) == 2:
            self.log("df", 0)
            self.log("cf", 0)
            return {"df": 0, "cf": 0}
        results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=self.args.label_file,
        )
        self.log("df", results["sent-detect-f1"])
        self.log("cf", results["sent-correct-f1"])
        for key, value in results.items():
            if 'acc' not in key:
                print(f'{key}: {value:.2f}')
        print()
        return {"df": results["sent-detect-f1"], "cf": results["sent-correct-f1"]}

    def train_dataloader(self) -> DataLoader:
        name = "train_all"

        dataset = Dynaimic_CSCDataset(
            data_path=os.path.join(self.args.data_dir, name),
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = dataset.tokenizer
        if not hasattr(self, 'func_get_sent_pinyin_ids'):
            self.func_get_sent_pinyin_ids = dataset.convert_sentence_to_pinyin_ids

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_for_train_dynamic_pron_loss, fill_values=[0, 0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        dataset = TestCSCDataset(
            data_path=f'data/test.sighan{self.args.val_sighan}.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        print('dev dataset', len(dataset))
        self.tokenizer = dataset.tokenizer

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,#self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = TestCSCDataset(
            data_path=f'data/test.sighan{self.args.val_sighan}.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader
    
    def updateGateMetric(self, tp, fn, fp, tn, whole):
        if hasattr(self, 'dic_gate') == False:
            self.dic_gate = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0, 'whole': 0}
        self.dic_gate['TP'] += tp
        self.dic_gate['FN'] += fn
        self.dic_gate['FP'] += fp
        self.dic_gate['TN'] += tn
        self.dic_gate['whole'] += whole
        return
    
    def getStepGateMetric(self, input_ids, pinyin_ids, labels, src='', tgt=''):
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        B, T = input_ids.shape
        pinyin_ids = pinyin_ids.view(B, T, 8)
        
        if self.args.model_type in ['lm']:
            tp, fn, fp, tn = 0, 0, 0, 0

        elif self.args.model_type in ['gate_copy_retr']:
            with torch.no_grad():
                attention_mask = (input_ids != 0).long()
                outputs_x2 = self.model.bert2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    output_hidden_states=True,
                )
                encoded_x2 = outputs_x2[0]  # (B, T, H)
                prediction_scores = self.model.cls2(encoded_x2)[0]  # (B, T, V)

            V = prediction_scores.shape[2]
            device = encoded_x2.device
        
            with torch.no_grad():
                gate_inputs = encoded_x2
                gate_scores = self.model.net_gate(gate_inputs)  # (B, T, 2)
                copy_gate_scores = self.model.net_copy_gate(gate_inputs)  # (B, T, 2)
            gate_probs = torch.softmax(gate_scores, dim=2)  # (B, T, 2)
            masks_retrieve = gate_probs[:, :, 0] < 0.5  # (B, T)
            label_masks_retrieve = input_ids != labels
            gate_pred = masks_retrieve.masked_select(mask.bool())
            gate_gold = label_masks_retrieve.masked_select(mask.bool())
            tp = ((gate_pred == 1) & (gate_gold == 1)).sum().item() 
            fn = ((gate_pred == 0) & (gate_gold == 1)).sum().item()
            fp = ((gate_pred == 1) & (gate_gold == 0)).sum().item()
            tn = ((gate_pred == 0) & (gate_gold == 0)).sum().item()
            whole = 1 
            for i in range(gate_gold.shape[0]):
                if gate_gold[i] == 1 and gate_pred[i] == 0:
                    whole = 0
                    break

        return tp, fn, fp, tn, whole

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dist = 0 if '13' in self.args.label_file else 1

        if self.args.name_retrieve == 'none':
            input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tgts, tokens_size = batch
            predict_labels = self.getPredictLabels(input_ids, pinyin_ids)
            tp, fn, fp, tn = self.getStepGateMetric(input_ids, pinyin_ids, labels, srcs[0], tgts[0])
            self.updateGateMetric(tp, fn, fp, tn)
            print(self.dic_gate)
            print()

            decoded_pred = self.tokenizer.decode_batch(predict_labels.cpu().tolist(), skip_special_tokens=True)
            preds = [''.join(decoded_one.split(' ')) for decoded_one in decoded_pred]
            src, tgt, pred = srcs[0], tgts[0], preds[0]
            flag_right = int(pred == tgt)
            flag_need = int(src != tgt)
            flag_change = int(src != pred)
            if flag_right == 0:
                print(f'modify No.1')
                print(f'  src : {src}')
                print(f'  tgt : {tgt}')
                print(f'  pred: {pred}')
                print(f'  right={flag_right}, need={flag_need}, change={flag_change}')
                print()

            if '13' in self.args.label_file:
                predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                    input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
            
            pre_predict_labels = predict_labels
            for _ in range(1):
                record_index = []
                for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                    if a != b:
                        record_index.append(i)
                
                input_ids[0, 1:-1] = predict_labels[0, 1:-1]
                sent, new_pinyin_ids = decode_sentence_and_get_pinyinids(input_ids[0, 1:-1].cpu().numpy().tolist())
                if new_pinyin_ids.shape[1] == input_ids.shape[1]:
                    pinyin_ids = new_pinyin_ids
                pinyin_ids = pinyin_ids.to(input_ids.device)
                predict_labels = self.getPredictLabels(input_ids, pinyin_ids)

                for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                    if a != b and any([abs(i - x) <= dist for x in record_index]):
                        1
                    else:
                        1
                        predict_labels[0, i+1] = input_ids[0, i+1]
                if '13' in self.args.label_file:
                    predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                        input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
            return {
                "tgt_idx": labels.cpu(),
                "post_pred_idx": predict_labels.cpu(),
                "pred_idx": pre_predict_labels.cpu(),
                "id": ids,
                "src": srcs,
                "tokens_size": tokens_size,
            }

        elif self.args.name_retrieve == 'knn' and 'retr' in self.args.model_type:
            input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tgts, tokens_size = batch
            predict_labels = self.getPredictLabels(input_ids, pinyin_ids)
            tp, fn, fp, tn, whole = self.getStepGateMetric(input_ids, pinyin_ids, labels, srcs[0], tgts[0])
            self.updateGateMetric(tp, fn, fp, tn, whole)
            print(self.dic_gate)
            print()
                
            if '13' in self.args.label_file:
                predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                    input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
            
            pre_predict_labels = predict_labels
            raw_input_ids = input_ids
            for i_modify in range(1):
                record_index = []
                for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                    if a != b:
                        record_index.append(i)
                
                input_ids[0, 1:-1] = predict_labels[0, 1:-1]
                sent, new_pinyin_ids = decode_sentence_and_get_pinyinids(input_ids[0, 1:-1].cpu().numpy().tolist())
                if new_pinyin_ids.shape[1] == input_ids.shape[1]:
                    pinyin_ids = new_pinyin_ids
                pinyin_ids = pinyin_ids.to(input_ids.device)
                predict_labels = self.getPredictLabels(input_ids, pinyin_ids)
                
                for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                    if a != b and any([abs(i - x) <= dist for x in record_index]):
                        1
                    else:
                        1
                        predict_labels[0, i+1] = input_ids[0, i+1]
                        
                if '13' in self.args.label_file:
                    predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                        input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
            
            return {
                "tgt_idx": labels.cpu(),
                "post_pred_idx": predict_labels.cpu(),
                "pred_idx": pre_predict_labels.cpu(),
                "id": ids,
                "src": srcs,
                "tokens_size": tokens_size,
            }


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--ckpt_path", default=None, type=str, help="resume_from_checkpoint")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default="data/test.sighan15.lbl.tsv", type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--model_type", default='gate_copy', type=str)

    parser.add_argument("--workers", type=int, default=8, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--save_topk", default=5, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default="train", type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.1, type=float, help="warmup proporation")
    parser.add_argument("--gamma", default=1, type=float, help="phonetic loss weight")

    parser.add_argument("--name_retrieve", default='knn', type=str)
    parser.add_argument("--dstore_dir", default='dstore', type=str)
    parser.add_argument("--dstore_maxsize", type=int, default=23591250)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--num_centroid", type=int, default=4096)
    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    val_sighan = '15'
    if '14' in args.label_file:
        val_sighan = '14'
    elif '13'in args.label_file:
        val_sighan = '13'
    args.val_sighan = val_sighan
    args.ngpu = len(args.gpus.strip(',').split(','))

    model = CSCTask(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoint"),
        filename="{epoch}-{df:.1f}-{cf:.1f}",
        save_top_k=args.save_topk,
        monitor="cf",
        mode="max",
    )
    logger = TensorBoardLogger(save_dir=args.save_path, name="log")

    # save args
    if not os.path.exists(os.path.join(args.save_path, "checkpoint")):
        os.mkdir(os.path.join(args.save_path, "checkpoint"))
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )
    trainer.fit(model)
    return

if __name__ == "__main__":
    freeze_support()
    main()
