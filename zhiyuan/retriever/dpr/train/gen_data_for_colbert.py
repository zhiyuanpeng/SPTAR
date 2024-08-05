"""
copy from train_sbert.py, gen train, dev data for colbert
"""
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join
import math
import sys
import pathlib, os, csv, random
from tqdm import tqdm
import json
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
parser.add_argument('--weak_num', required=False, default="100k", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#
colbert_dir = join(cwd, "zhiyuan", "retriever", "col_bert")
colbert_data_dir = join(colbert_dir, "data")
colbert_dataset_dir = join(colbert_data_dir, "datasets")
colbert_saveto_dir = join(colbert_dataset_dir, args.dataset_name, args.exp_name)
os.makedirs(colbert_saveto_dir, exist_ok=True)
#### Provide model save path
#### Just some code to print debug information to stdout
fh = logging.FileHandler(join(colbert_saveto_dir, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])
#### /print debug information to stdout
#### Provide the data_path where nfcorpus has been downloaded and unzipped
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

train_collection_path = join(colbert_saveto_dir, "train", "collection.tsv")
train_queries_path = join(colbert_saveto_dir, "train", "queries.tsv")
train_triples_path = join(colbert_saveto_dir, "train", "triples.jsonl")
os.makedirs("/".join(train_collection_path.split("/")[:-1]), exist_ok=True)
os.makedirs("/".join(train_queries_path.split("/")[:-1]), exist_ok=True)
os.makedirs("/".join(train_triples_path.split("/")[:-1]), exist_ok=True)

dev_collection_path = join(colbert_saveto_dir, "dev", "collection.tsv")
dev_queries_path = join(colbert_saveto_dir, "dev", "queries.tsv")
dev_qrels_path = join(colbert_saveto_dir, "dev", "qrels.jsonl")
os.makedirs("/".join(dev_collection_path.split("/")[:-1]), exist_ok=True)
os.makedirs("/".join(dev_queries_path.split("/")[:-1]), exist_ok=True)
os.makedirs("/".join(dev_qrels_path.split("/")[:-1]), exist_ok=True)
#### Create output directories for collection and queries

def preprocess(text):
    return text.replace("\r", " ").replace("\t", " ").replace("\n", " ")

def sample_negative(corpus_ids, qrels, seed=4589):
    triples = []
    random.seed(seed)
    for q_id, pids_dict in qrels.items():
        pids = set(pids_dict.keys())
        sample_num = len(pids)*2
        sampled_nids = random.sample(corpus_ids, sample_num)
        for pid in pids:
            if pid in sampled_nids:
                sampled_nids.remove(pid)
        assert len(sampled_nids) >= len(pids)
        sampled_nids = sampled_nids[:len(pids)]
        for pid, nid in zip(pids, sampled_nids):
            triples.append([q_id, pid, nid])
    return triples
    
train_corpus_ids = list(corpus)
logging.info("Preprocessing Train Corpus and Saving to {} ...".format(train_collection_path))
with open(train_collection_path, 'w') as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for doc_id in tqdm(train_corpus_ids, total=len(train_corpus_ids)):
        doc = corpus[doc_id]
        writer.writerow([doc_id,(preprocess(doc.get("title", "")) + "|" + preprocess(doc.get("text", ""))).strip()])
        # writer.writerow([doc_id,(preprocess(doc.get("text", ""))).strip(),preprocess(doc.get("title", "")).strip()])

logging.info("Preprocessing Train Queries and Saving to {} ...".format(train_queries_path))
with open(train_queries_path, 'w') as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for qid, query in tqdm(queries.items(), total=len(queries)):
        writer.writerow([qid, query])
#
train_triples = sample_negative(train_corpus_ids, qrels)
logging.info("Preprocessing Train Triples and Saving to {} ...".format(train_triples_path))
with open(train_triples_path, 'w') as fIn:
    for t in train_triples:
        json.dump(t, fIn)
        fIn.write("\n")
#
dev_corpus_ids = list(dev_corpus)
logging.info("Preprocessing Dev Corpus and Saving to {} ...".format(dev_collection_path))
with open(dev_collection_path, 'w') as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for doc_id in tqdm(dev_corpus_ids, total=len(dev_corpus_ids)):
        doc = dev_corpus[doc_id]
        writer.writerow([doc_id,(preprocess(doc.get("title", "")) + "|" + preprocess(doc.get("text", ""))).strip()])
        # writer.writerow([doc_id,(preprocess(doc.get("text", ""))).strip(),preprocess(doc.get("title", "")).strip()])

logging.info("Preprocessing Dev Queries and Saving to {} ...".format(dev_queries_path))
with open(dev_queries_path, 'w') as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for qid, query in tqdm(dev_queries.items(), total=len(dev_queries)):
        writer.writerow([qid, query])

# dev_triples = sample_negative(dev_corpus_ids, dev_qrels)
ori_dev_qrels_path = join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
logging.info("Preprocessing Train Triples and Saving to {} ...".format(dev_qrels_path))
with open(dev_qrels_path, 'w') as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    reader = csv.reader(open(ori_dev_qrels_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        writer.writerow([query_id, 0, corpus_id, 1])

# add the test data, for colbert, dev is not used as there is no evaluation during the training, for fiqa, load the original test q and corpus to colbert folder. For mamarco, load the original, process the original corpus to colbert format. manually add trec2019 before trec dl 2019 queries and add trec2020 before trec dl 2020 queries. append trec dl 2019 200 test queries and trec dl 2020 200 test queries to msmarco dev queries as test queries. For both fiqa and msmarco, original test qrels will be loaded for evalution.

    
    

