#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   tasb_teacher.py
@Time    :   2024/06/07 18:30:53
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   load the ColBERT and BERT_CAT model for teachers.
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models as beir_models
import pathlib, os
import logging
import argparse
from os.path import join
import math
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.cluster import KMeans
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from weak_data_loader import WeakDataLoader
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
model_name = "tasb"
#### Provide the data_path where nfcorpus has been downloaded and unzipped
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
    output_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"tasb_q_{args.train_num}_clustring.txt")
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
    output_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"tasb_q_{args.exp_name}_clustring.txt")

model = SentenceTransformer("msmarco-bert-base-dot-v5")
q_texts, q_ids = [], []
for q_id, q_text in queries.items():
    q_texts.append(q_text)
    q_ids.append(q_id)
query_embeddings = model.encode(q_texts)
cluster_num = math.ceil(len(q_texts) / 200)

# Perform clustering
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(query_embeddings)

# Create a list of empty lists to hold the query ids for each cluster
clusters = [[] for _ in range(cluster_num)]

# Assign each query id to a cluster
for q_id, cluster_id in zip(q_ids, kmeans.labels_):
    clusters[cluster_id].append(q_id)

# Write clusters to the output file
with open(output_file, 'w') as f:
    for cluster in clusters:
        f.write(','.join(map(str, cluster)) + '\n')


