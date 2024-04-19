import os
import time 
import faiss
import argparse
import numpy as np
from pytorch_lightning import Trainer


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--save_path", default=os.path.join('..', 'dstore'), type=str)
    parser.add_argument("--name_record_maxsize", default='latest_maxsize.txt', type=str)
    parser.add_argument("--name_retrieve_prepare", default='knn', type=str)
    parser.add_argument("--text_type", default='both', type=str)

    parser.add_argument("--dim_hidden", default=768, type=int)
    parser.add_argument("--num_bit", default=16, type=int)

    parser.add_argument('--num_bit_per_index', type=int, default=8)
    parser.add_argument('--num_centroid', type=int, default=4096)
    parser.add_argument('--num_probe', type=int, default=32)
    parser.add_argument('--num_sub_quantizer', type=int, default=64) 
    parser.add_argument('--num_key_per_add', type=int, default=500000) 
    return parser


if __name__ == '__main__':
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    path_record_maxsize = os.path.join(args.save_path, args.name_record_maxsize)
    with open(path_record_maxsize, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        dstore_maxsize = int(line)
    ngpu = len(args.gpus.strip(',').split(','))
    
    # dstore
    path_dstore_keys = os.path.join(args.save_path, f'{args.name_retrieve_prepare}_{args.text_type}_keys_b{args.num_bit}_ds{dstore_maxsize}.npy')
    path_dstore_values = os.path.join(args.save_path, f'{args.name_retrieve_prepare}_{args.text_type}_values_b{args.num_bit}_ds{dstore_maxsize}.npy')

    dtype_float = np.float32 if args.num_bit == 32 else np.float16
    dtype_long = np.int64
    dstore_keys = np.memmap(path_dstore_keys, dtype=dtype_float, mode='r', shape=(dstore_maxsize, args.dim_hidden))
    dstore_values = np.memmap(path_dstore_values, dtype=dtype_long, mode='r', shape=(dstore_maxsize))

    name_index = f'{args.name_retrieve_prepare}_{args.text_type}_index_b{args.num_bit}_c{args.num_centroid}_p{args.num_probe}_s{args.num_sub_quantizer}_ds{dstore_maxsize}'
    path_index_train = os.path.join(args.save_path, f'{name_index}_train')
    path_index_add = os.path.join(args.save_path, f'{name_index}_add')
    
    resource = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatIP(args.dim_hidden)
    if not os.path.exists(path_index_train):
        index = faiss.IndexIVFPQ(quantizer, args.dim_hidden, args.num_centroid, args.num_sub_quantizer, args.num_bit_per_index, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = args.num_probe

        print('[debug] put index to gpu')
        gpu_index = index

        print('[debug] train index')
        inds = np.random.choice(np.arange(
            dstore_values.shape[0]), size=[min(1000000, dstore_values.shape[0])], replace=False)
        inds = np.sort(inds)
        start_train = time.time()
        dstore_keys_train = dstore_keys[inds].astype(np.float32)
        faiss.normalize_L2(dstore_keys_train)
        gpu_index.train(dstore_keys_train)
        print(f'[debug] train took {time.time() - start_train:.0f} s')

        print('[debug] write index train')
        start_write = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), path_index_train)
        print(f'[debug] write took {time.time() - start_write:.0f} s')

    print(f'[debug] read index')
    index = faiss.read_index(path_index_train)
    cloner_options = faiss.GpuClonerOptions()
    cloner_options.useFloat16 = args.num_bit == 16
    clustering_index = faiss.index_cpu_to_all_gpus(quantizer, ngpu=ngpu)
    gpu_index = index
    gpu_index.clustering_index = clustering_index

    start_add = time.time()
    i, j = 0, 0
    while i < dstore_maxsize:
        j = min(dstore_maxsize, i + args.num_key_per_add)
        dstore_keys_add = dstore_keys[i:j].copy().astype(np.float32)
        faiss.normalize_L2(dstore_keys_add)
        gpu_index.add_with_ids(dstore_keys_add, np.arange(i, j))
        i += args.num_key_per_add

        if (i % 1000000) == 0:
            print(f'[debug] add {i} keys')
            print('[debug] write index add')
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), path_index_add)

    print(f'[debug] add total {j} keys')
    print(f'[debug] add took {time.time() - start_add:.0f} s')
    print('[debug] write index add')
    start_write = time.time()
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), path_index_add)
    print(f'[debug] write took {time.time() - start_write:.0f} s')
    print(f'[debug] {path_dstore_keys} can remove')
