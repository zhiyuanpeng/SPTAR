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

Usage: python zhiyuan/filter/bm25anserini_split.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import pathlib, os, json
import logging
import requests
import random
import sys
from os.path import join
import argparse
import math
from tqdm import tqdm
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
from data_process import read_weak_json
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = "/home/vn55kb9/proj/LLMsAgumentedIR/xuyang/data"

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="100k", type=str)
parser.add_argument('--port', required=False, default=8000, type=int)
parser.add_argument('--overwritejsonl', action='store_true')
parser.add_argument('--reindex', action='store_true')
parser.add_argument('--exp_name', required=False, default="llama_7b_none_5000", type=str)
parser.add_argument('--topk', required=False, default=10, type=int)
args = parser.parse_args()
#### Provide model save path
log_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), args.dataset_name, args.weak_num, str(args.topk))
pyserini_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), args.dataset_name)
os.makedirs(log_path, exist_ok=True)
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(log_path, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])

#
gen_file_dir = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", "100k")
weak_q_path = join(gen_file_dir, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
weak_qrel_path = join(gen_file_dir, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, "corpus.jsonl"), query_file=weak_q_path, qrels_file=weak_qrel_path).load_custom()
#### Convert BEIR corpus to Pyserini Format #####
pyserini_jsonl = "pyserini.jsonl"
if args.overwritejsonl or not os.path.exists(os.path.join(pyserini_path, pyserini_jsonl)):
    print(f"write to {pyserini_jsonl}...")                                        
    with open(os.path.join(pyserini_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
docker_beir_pyserini = f"http://127.0.0.1:{args.port}"

if args.reindex:
    #### Upload Multipart-encoded files ####
    with open(os.path.join(pyserini_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    #### Index documents to Pyserini #####
    index_name = f"beir/{args.dataset_name}" # beir/scifact
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

#### Retrieve documents from Pyserini #####
retriever = EvaluateRetrieval()
qids = list(queries)
query_texts = [queries[qid] for qid in qids]
chunk_size = 5000
chunk_num = math.ceil(len(qids)/chunk_size)

raw_weak_queries = read_weak_json(weak_q_path)
filtered_num = 0
## write filtered query and qrel files to ...
filtered_weak_q_path = join(gen_file_dir, f"weak_queries_{args.train_num}_{args.exp_name}_filtered_{args.topk}.jsonl")
filtered_weak_qrel_path = join(gen_file_dir, f"weak_train_{args.train_num}_{args.exp_name}_filtered_{args.topk}.tsv")

if os.path.exists(filtered_weak_q_path):
    os.remove(filtered_weak_q_path)
    logging.info(f"del old {filtered_weak_q_path}")

if os.path.exists(filtered_weak_qrel_path):
    os.remove(filtered_weak_qrel_path)
    logging.info(f"del old {filtered_weak_qrel_path}")
f1 = open(filtered_weak_qrel_path, "w+")
line = f"query-id\tcorpus-id\tscore\n"
f1.write(line)
f2 = open(filtered_weak_q_path, "w+")
for i in tqdm(range(chunk_num)):
    if i != chunk_num - 1:
        payload = {"queries": query_texts[i*chunk_size: (i+1)*chunk_size], "qids": qids[i*chunk_size: (i+1)*chunk_size], "k": args.topk}
        raw_weak_queries_chunk = {}
        qrels_chunk = {}
        for qq_id in qids[i*chunk_size: (i+1)*chunk_size]:
            raw_weak_queries_chunk[qq_id] = raw_weak_queries[qq_id]
            qrels_chunk[qq_id] = qrels[qq_id]
    else:
        payload = {"queries": query_texts[i*chunk_size:], "qids": qids[i*chunk_size:], "k": args.topk}
        raw_weak_queries_chunk = {}
        for qq_id in qids[i*chunk_size:]:
            raw_weak_queries_chunk[qq_id] = raw_weak_queries[qq_id]
            qrels_chunk[qq_id] = qrels[qq_id]
    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    #### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
    # results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

    #### Check if query_id is in results i.e. remove it from docs incase if it appears ####
    #### Quite Important for ArguAna and Quora ####
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    if i == 0:
        logging.info("Before filter:")
        logging.info(f"There are {len(queries)} weak queries")
    
    # iterate each q and retrieved topk doc list pair
    for q_id, topk_doc_ids in results.items():
        topk_doc_ids = [(ss, ii) for ii, ss in topk_doc_ids.items()]
        sorted(topk_doc_ids, reverse=True)
        topk_doc_ids = topk_doc_ids[:args.topk]
        topk_doc_ids = [ii for (_, ii) in topk_doc_ids]
        # extract the weak doc id, there is only one weak doc for each weak query
        weak_doc_id = list(qrels_chunk[q_id].keys())[0]
        # if weak id not in topk retrieved doc list, remove it from the weak doc id list to make the weak doc more cleaner
        if weak_doc_id not in topk_doc_ids:
            # del qrels[q_id][weak_doc_id]
        # if not weak doc id shown in topk bm25 retrieved doc list, then this query is not qualified
        # if len(qrels[q_id]) == 0:
            del raw_weak_queries_chunk[q_id]
            del qrels_chunk[q_id]

    filtered_num += len(raw_weak_queries_chunk)
    
    for k, vals in qrels_chunk.items():
        for docid, docscore in vals.items():
            line = f"{k}\t{docid}\t1\n"
            f1.write(line)

    filtered_weak_q = list(raw_weak_queries_chunk.values())
    
    for qualified_weak_q in filtered_weak_q:
        json.dump(qualified_weak_q, f2)
        f2.write("\n")

f1.close()
f2.close()
logging.info("After filter:")
logging.info(f"There are {filtered_num} weak queries")