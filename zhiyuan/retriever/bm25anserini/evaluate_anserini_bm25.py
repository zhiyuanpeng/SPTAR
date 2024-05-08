"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull beir/pyserini-fastapi 
2. docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
3. docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest
Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.

Usage: python evaluate_anserini_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from zhiyuan.data_process import load_dl
import pathlib, os, json
import logging
import requests
import random

from os.path import join
import argparse

####
cwd = os.getcwd()
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--skip_index', action='store_true')
parser.add_argument('--skip_retrieval', action='store_true')
args = parser.parse_args()
#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.dataset_name)
os.makedirs(model_save_path, exist_ok=True)
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(model_save_path, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

# corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, "corpus.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "test.tsv")).load_custom()
if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
elif args.dataset_name == "dl2019":
    queries, qrels, qrels_binary = load_dl(join(beir_dir, "TREC_DL_2019"))
    if not args.skip_index:
        corpus, _, _ = GenericDataLoader(join(beir_dir, "msmarco")).load(split="dev")
elif args.dataset_name == "dl2020":
    queries, qrels, qrels_binary = load_dl(join(beir_dir, "TREC_DL_2020"))
    if not args.skip_index:
        corpus, _, _ = GenericDataLoader(join(beir_dir, "msmarco")).load(split="dev")
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
port_dict = {"msmarco": 8000, "fiqa": 8002, "dl2019": 8000, "dl2020": 8000}
docker_beir_pyserini = f"http://127.0.0.1:{port_dict[args.dataset_name]}"
if args.dataset_name == "msmarco" or args.dataset_name == "dl2019" or args.dataset_name == "dl2020":
    index_name = f"beir/msmarco" # beir/scifact
else:
    index_name = f"beir/{args.dataset_name}"
if not args.skip_index:
    #### Convert BEIR corpus to Pyserini Format #####
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(model_save_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')
    #### Upload Multipart-encoded files ####
    with open(os.path.join(model_save_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)
    #### Index documents to Pyserini #####
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})
if not args.skip_retrieval:
    #### Retrieve documents from Pyserini #####
    retriever = EvaluateRetrieval(k_values=[1,3,5,10, 1000])
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    #### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
    # results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

    #### Check if query_id is in results i.e. remove it from docs incase if it appears ####
    #### Quite Important for ArguAna and Quora ####
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, map, recall, _ = retriever.evaluate(qrels, results, retriever.k_values)
    if args.dataset_name == "dl2019" or args.dataset_name == "dl2020":
        _, map, recall, _ = retriever.evaluate(qrels_binary, results, retriever.k_values)
    else:
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        for eval in [mrr]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))
    for eval in [ndcg, map, recall]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))
    #### Retrieval Example ####
    # query_id, scores_dict = random.choice(list(results.items()))
    # logging.info("Query : %s\n" % queries[query_id])

    # scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    # for rank in range(10):
    #     doc_id = scores[rank][0]
    #     logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))