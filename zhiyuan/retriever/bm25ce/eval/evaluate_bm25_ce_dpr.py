"""
copy from evaluate_bm25_ce_reranking.py, use trained dpr bert replace the default sequenceclassification model
"""
"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull beir/pyserini-fastapi 
2. docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
4. docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest

Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.

Usage: python evaluate_anserini_bm25.py
"""
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from beir.retrieval import models
import pathlib, os, json
import logging
import requests
import random
from time import time
from os.path import join
import argparse
import sys
####
cwd = os.getcwd()
zhiyuan_dir = join(cwd, "zhiyuan")
if zhiyuan_dir not in sys.path:
    sys.path.append(zhiyuan_dir)
from data_process import load_dl, merge_queries, extract_results
from DenseRetrievalExactSearchBM25 import DenseRetrievalExactSearchBM25 as DRES
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")
dpr_dir = join(cwd, "zhiyuan", "retriever", "dpr")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="fiqa", type=str)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--dpr_v', required=False, default="v1", choices=["v1", "v2"], type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
parser.add_argument('--topk', required=False, default=1000, type=int)
args = parser.parse_args()
#### Provide model save path
model_name = "bert-base-uncased" 
log_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), f"top_{args.topk}", "{}-{}-{}".format(model_name, args.dpr_v, args.dataset_name))
os.makedirs(log_path, exist_ok=True)
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(log_path, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])

model_save_path = os.path.join(dpr_dir, "train", "output", args.exp_name, str(args.train_num), "{}-{}-{}".format(model_name, args.dpr_v, args.dataset_name))

if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
    queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
    queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")
##################################################
#### (1) RETRIEVE Top-100 docs using BM25 Pyserini
##################################################
port_dict = {"msmarco": 8000, "fiqa": 8002}
docker_beir_pyserini = f"http://127.0.0.1:{port_dict[args.dataset_name]}"
if args.dataset_name == "msmarco":
    queries = merge_queries(queries, queries_19, queries_20)
qids = list(queries)
query_texts = [queries[qid] for qid in qids]
payload = {"queries": query_texts, "qids": qids, "k": 1000}
#### Retrieve pyserini results (format of results is identical to qrels)
results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
#### Check if query_id is in results i.e. remove it from docs incase if it appears ####
#### Quite Important for ArguAna and Quora ####
# for query_id in results:
#     if query_id in results[query_id]:
#         results[query_id].pop(query_id, None)
# # manually extract top 100
results_topk = {}
for q_id, topk_doc_ids in results.items():
    topk_doc_ids = [(ss, ii) for ii, ss in topk_doc_ids.items()]
    sorted(topk_doc_ids, reverse=True)
    topk_doc_ids = topk_doc_ids[:args.topk]
    topk_doc_ids = [ii for (ss, ii) in topk_doc_ids]
    results_topk[q_id] = topk_doc_ids

################################################
#### (2) RERANK Top-100 docs using Cross-Encoder
################################################
trained_model = models.SentenceBERT(model_save_path)
model = DRES(trained_model, batch_size=256)

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = model.search(corpus, queries, results_topk, args.topk, score_function="cos_sim", return_sorted=True)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

tobe_eval = {}
if args.dataset_name == "msmarco":
    results, results_19, results_20 = extract_results(results)
    tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
    tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
tobe_eval[args.dataset_name] = (qrels, results, "pad")

retriever = EvaluateRetrieval(k_values=[1,3,5,10,100,300, 500, 1000], score_function="cos_sim")

for dataset_name in tobe_eval.keys():
    qrels, results, qrels_binary = tobe_eval[dataset_name]
    logging.info("Retriever evaluation for dataset {}".format(dataset_name))
    ndcg, map, recall, _, score_per_query= retriever.evaluate(qrels, results, retriever.k_values)
    if dataset_name == "dl2019" or dataset_name == "dl2020":
        _, map, recall, _, score_per_query_override = retriever.evaluate(qrels_binary, results, retriever.k_values)
        for key in score_per_query.keys():
            if "MAP" in key:
                score_per_query[key] = score_per_query_override[key]
            if "Recall" in key:
                score_per_query[key] = score_per_query_override[key]
    else:
        mrr, mrr_score = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        for key in mrr_score.keys():
            score_per_query[key] = mrr_score[key]
    for eval in [ndcg, map, recall]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))
    with open(join(log_path, f"{dataset_name}.json"), "w") as f:
        json.dump(score_per_query, f)