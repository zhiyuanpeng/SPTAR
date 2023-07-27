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
####
cwd = os.getcwd()
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
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
model_name = "bert-base-uncased" 
model_save_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "train", "output", args.exp_name, str(args.train_num), "{}-{}-{}".format(model_name, args.dpr_v, args.dataset_name))
os.makedirs(model_save_path, exist_ok=True)
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(model_save_path, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])
#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

# corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, "corpus_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "test.tsv")).load_custom()

if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

# model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=256, corpus_chunk_size=512*9999)
model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
retriever = EvaluateRetrieval(model, k_values=[1,3,5,10], score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

for eval in [mrr, recall_cap, hole]:
    logging.info("\n")
    for k in eval.keys():
        logging.info("{}: {:.4f}".format(k, eval[k]))

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))