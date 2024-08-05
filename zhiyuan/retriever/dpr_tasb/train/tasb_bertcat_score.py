#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   tasb_teacher.py
@Time    :   2024/06/07 18:30:53
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   load the ColBERT and BERT_CAT model for teachers.
'''
from sentence_transformers import losses, models, SentenceTransformer, CrossEncoder
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
import random
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
parser.add_argument('--num_epochs', required=False, default=20, type=int)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--product', required=False, default="cosine", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
model_name = "tasb"
#### Provide the data_path where nfcorpus has been downloaded and unzipped
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
    output_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"tasb_{args.train_num}_teacher_scores.txt")
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
    output_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"tasb_{args.exp_name}_teacher_scores.txt")

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
# usage
# scores = model.predict(
#     [("Query", "Paragraph1"), ("Query", "Paragraph2"), ("Query", "Paragraph3")]
# )

# output format: scorpos<tab>scoreneg<tab>query<tab>docpos<tab>docneg

# Open the output file
with open(output_file, 'w') as f:
    # Iterate over each query in qrels
    for query_id, pos_docs in tqdm(qrels.items()):
        # Get the positive document IDs
        pos_doc_ids = list(pos_docs.keys())
        # Get the negative document IDs
        neg_doc_ids = [doc_id for doc_id in corpus.keys() if doc_id not in pos_doc_ids]
        # Sample 20 negative documents
        neg_doc_ids = random.sample(neg_doc_ids, 20)
        # Get the query text
        query_text = queries[query_id]
        # Get the positive and negative document texts
        pos_doc_texts = [corpus[doc_id]['text'] for doc_id in pos_doc_ids]
        neg_doc_texts = [corpus[doc_id]['text'] for doc_id in neg_doc_ids]
        # Predict the scores for the positive and negative documents
        pos_scores = model.predict([(query_text, doc_text) for doc_text in pos_doc_texts])
        neg_scores = model.predict([(query_text, doc_text) for doc_text in neg_doc_texts])
        # Pair each positive document with each negative document and save the scores, query ID, positive document ID, and negative document ID to the output file
        for pos_score, pos_doc_id in zip(pos_scores, pos_doc_ids):
            for neg_score, neg_doc_id in zip(neg_scores, neg_doc_ids):
                f.write(f"{pos_score}\t{neg_score}\t{query_id}\t{pos_doc_id}\t{neg_doc_id}\n")
