import os
import sys
import time 
import faiss
import torch
import numpy as np
import faiss.contrib.torch_utils

class KnnRetriever:
    def __init__(
        self,
        args_retriever,
        output_file=None
    ):
        self.args = args_retriever
        self.index = self.getIndex()
        self.output_file = output_file
        if output_file:
            with open(output_file, 'w') as o:
                pass

    def getIndex(self):
        args = self.args
        dim_hidden = args.dim_hidden
        dstore_maxsize = args.dstore_maxsize
        path_dstore_keys = args.path_dstore_keys
        path_dstore_values = args.path_dstore_values
        path_index = args.path_index
        ngpu = args.ngpu
        num_bit = args.num_bit
        num_probe = args.num_probe

        print(f'[debug] read index from {path_index}')
        start_read = time.time()
        quantizer = faiss.IndexFlatIP(dim_hidden)
        index = faiss.read_index(path_index, faiss.IO_FLAG_ONDISK_SAME_DIR)
        gpu_index = index
        gpu_index.nprobe = num_probe
        print(f'[debug] read took {time.time() - start_read:.0f} s')

        print(f'[debug] load dstore fp{num_bit}: ({dstore_maxsize}, {dim_hidden})')
        dtype_float = np.float32 if num_bit == 32 else np.float16
        dtype_long = np.int64
        dstore_keys = np.memmap(path_dstore_keys, dtype=dtype_float, mode='r', shape=(dstore_maxsize, dim_hidden))
        dstore_values = np.memmap(path_dstore_values, dtype=dtype_long, mode='r', shape=(dstore_maxsize,))
        self.dstore_keys = dstore_keys
        self.dstore_values = dstore_values
        return gpu_index
    
    def getProbsRetrieve(self, dic_retrieve):
        args = self.args
        temperature = args.temperature
        vocab_size = args.vocab_size

        dists = dic_retrieve['dists']
        values = dic_retrieve['values'].unsqueeze(3)  # (B, T, K, 1)
        B, T, num_k = dists.shape
        device = dists.device
        scores = -dists / temperature  # (B, T, K)
        scores = torch.softmax(scores, dim=2).unsqueeze(3)  # (B, T, K, 1)
        probs_retrieve = torch.zeros(B, T, num_k, vocab_size).to(device)  # (B, T, K, V)
        probs_retrieve = probs_retrieve.scatter(dim=3, index=values, src=scores)
        probs_retrieve = probs_retrieve.sum(dim=-2)  # (B, T, V)
        return probs_retrieve
    
    def getRetrieveOutputs(self, dic_retrieve):
        retrieve_dists = dic_retrieve['dists']  # (B, T, K)
        retrieve_inds = dic_retrieve['inds']  # (B, T, K)
        retrieve_embeds = dic_retrieve['keys']  # (B, T, K, H)
        retrieve_values = dic_retrieve['values']  # (B, T, K)

        retrieve_outputs = {
            'retrieve_dists': retrieve_dists,  # (B, T, K)
            'retrieve_inds': retrieve_inds,  # (B, T, K)
            'retrieve_embeds': retrieve_embeds,  # (B, T, K, H)
            'retrieve_values': retrieve_values,  # (B, T, K)
        }
        return retrieve_outputs
    
    def retrieve(
        self,
        encoded: torch.FloatTensor,
    ):
        r"""
        Args:
            encoded (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim_hidden)`)
        """
        args = self.args
        num_k = args.num_k
        B, T, H = encoded.shape
        device = encoded.device
        queries = encoded.detach().cpu().reshape(B * T, H).contiguous()
        dists, inds = self.index.search(queries, num_k)  # (B * T, K)
        keys = torch.from_numpy(self.dstore_keys[inds])
        values = torch.from_numpy(self.dstore_values[inds])

        f1 = lambda x: x.reshape(B, T, num_k).to(device)
        f2 = lambda x: x.reshape(B, T, num_k, H).to(device)
        dists = f1(dists)  # (B, T, K)
        inds = f1(inds)
        values = f1(values)
        keys = f2(keys)  # (B, T, K, H)
        dic_retrieve = {
            'dists': dists,
            'inds': inds, 
            'keys': keys,
            'values': values
        }
        return dic_retrieve


class KnnRetrieverArgs:
    def __init__(
        self,
        dstore_dir,
        dstore_maxsize,
        lamb,
        ngpu,
        temperature,
        decode_context=False,
        dim_hidden=768,
        num_bit=16,
        num_centroid=4096,
        num_k=3,
        num_probe=32,
        num_sub_quantizer=64,
        tokenizer=None,
        vocab_size=23236,
    ):
        name_retrieve = 'knn_both'
        name_index = f'{name_retrieve}_index_b{num_bit}_c{num_centroid}_p{num_probe}_s{num_sub_quantizer}_ds{dstore_maxsize}'

        self.decode_context = decode_context
        self.dim_hidden = dim_hidden
        self.dstore_maxsize = dstore_maxsize
        self.lamb = lamb
        self.ngpu = ngpu
        self.num_bit = num_bit
        self.num_k = num_k
        self.num_probe = num_probe
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        self.path_dstore_keys = os.path.join(dstore_dir, f'{name_retrieve}_keys_b{num_bit}_ds{dstore_maxsize}.npy')
        self.path_dstore_values = os.path.join(dstore_dir, f'{name_retrieve}_values_b{num_bit}_ds{dstore_maxsize}.npy')
        self.path_index = os.path.join(dstore_dir, f'{name_index}_add')
