from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import pathlib, os
import random
import argparse
from os.path import join
import sys
import json
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from data_process import load_dl, merge_queries, extract_results
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--dpr_v', required=False, default="v1", choices=["v1", "v2"], type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
model_name = "facebook/contriever"
model_save_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "train", "output", args.exp_name, str(args.train_num), "{}-{}-{}".format(model_name, args.dpr_v, args.dataset_name))
os.makedirs(model_save_path, exist_ok=True)
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(model_save_path, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])

model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,300,500,1000], score_function="cos_sim")

if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
    queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
    queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

# for testing, sampel 100k from corpus
# corpus = dict(random.sample(corpus.items(), 100000))
tobe_eval = {}
if args.dataset_name == "msmarco":
    ms_queries = merge_queries(queries, queries_19, queries_20)
    ms_results = retriever.retrieve(corpus, ms_queries)
    results, results_19, results_20 = extract_results(ms_results)
    tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
    tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
else:
    results = retriever.retrieve(corpus, queries)
tobe_eval[args.dataset_name] = (qrels, results, "pad")

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
    with open(join(model_save_path, f"{dataset_name}.json"), "w") as f:
        json.dump(score_per_query, f)

