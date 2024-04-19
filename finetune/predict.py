import os
import re
import time
import argparse
from pytorch_lightning import Trainer
from multiprocessing import freeze_support

from metrics.metric import Metric
from finetune.train import CSCTask
from retrieve.retrievers import KnnRetriever, KnnRetrieverArgs


def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--ckpt_path", type=str, help="ckpt file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default="data/test.sighan15.lbl.tsv", type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    parser.add_argument("--workers", type=int, default=8, help="num workers for dataloader")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")

    parser.add_argument("--model_type", type=str, default='copy')
    parser.add_argument("--name_retrieve", default='none', type=str)
    parser.add_argument("--dstore_dir", default='dstore', type=str)
    parser.add_argument("--dstore_maxsize", type=int, default=22045202)
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
    trainer = Trainer.from_argparse_args(args)
    t_start = time.time()
    output = trainer.predict(model=model, dataloaders=model.test_dataloader(), ckpt_path=args.ckpt_path)
    t_delta = time.time() - t_start
  
    metric = Metric(vocab_path=args.bert_path)
    pred_txt_path = os.path.join(args.save_path, "preds.txt")
    pred_lbl_path = os.path.join(args.save_path, "labels.txt")

    results = metric.metric(
        batches=output,
        pred_txt_path=pred_txt_path,
        pred_lbl_path=pred_lbl_path,
        label_path=args.label_file,
        should_remove_de=True if '13'in args.label_file else False
    )
    print('raw')
    for k, v in results.items():
        if 'acc' not in k:
            print(f'{k}: {v:.2f}')

    for ex in output:
        ex['pred_idx'] = ex['post_pred_idx']
    results = metric.metric(
        batches=output,
        pred_txt_path=pred_txt_path,
        pred_lbl_path=pred_lbl_path,
        label_path=args.label_file,
        should_remove_de=True if '13'in args.label_file else False
    )
    print('post')
    for k, v in results.items():
        if 'acc' not in k:
            print(f'{k}: {v:.2f}')
    print(f'time: {t_delta:.0f}')
    return


if __name__ == '__main__':
    freeze_support()
    main()
    