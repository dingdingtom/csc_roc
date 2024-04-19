## RoC

### Title

Retrieve-or-Copy: Enhancing Chinese Spelling Check Models with a Learnable Retrieval

### Requirements

CUDA 12.2

```bash
pip install -r requirements.txt
conda install -c pytorch faiss-gpu
```

### Data

[SIGHAN-data](https://github.com/jiahaozhenbang/SCOPE): the preprocessed data by SCOPE

[FPT-model](https://github.com/jiahaozhenbang/SCOPE): the further pre-trained model by SCOPE

[SCOPE-paper](https://aclanthology.org/2022.emnlp-main.287/)

### Structure

```
|__ data 
|__ data_process
|   |__ get_train_data.py
|__ datasets
|   |__ __init__.py
|   |__ bert_csc_dataset.py
|   |__ chinese_bert_dataset.py
|   |__ collate_function.py
|   |__ utils.py
|__ finetune
|   |__ predict.py
|   |__ train.py
|__ FPT
|__ metrics
|   |__ __init__.py
|   |__ metric.py
|   |__ metric_core.py
|   |__ remove_de.py
|__ models
|   |__ classifier.py
|   |__ fusion_embedding.py
|   |__ glyph_embedding.py
|   |__ modeling_glycebert.py
|   |__ modeling_multitask.py
|   |__ pinyin_embedding.py
|__ retrieve
|   |__ prepare_dstore.py
|   |__ retrievers.py
|   |__ train_dstore.py
|__ utils
|   |__ __init__.py
|   |__ random_seed.py
|__ 1.run_pretrain.sh
|__ 2.run_dstore.sh
|__ 3.run_train.sh
|__ 4.run_predict.sh
|__ requirements.txt
```

### Results

```bash
python data_process/get_train_data.py \
    --data_path data \
    --output_dir data
bash 1.run_pretrain.sh
bash 2.run_dstore.sh
bash 3.run_train.sh
bash 4.run_predict.sh
```

Note. Please rewrite `ckpt_path`, `val_sighan` during all stages. You can follow the Structure section above to save checkpoints and data.

Note. The SIGHAN13/14/15 should be trained individually using all training data to obtain the best results.

Note. The first step is to pretrain a base model for encoding and retrieving. The second step is to build the datastore using FAISS. The third step is to train. The fourth step is to test.

### Acknowledgement

The code is modified upon the released code of 2022 EMNLP paper "Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity".

We appreciate their open-sourcing such high-quality code and FPT model, which is very helpful to our research. We thank pytorch and pytorch-lightning for their wonderful training implementation. 

