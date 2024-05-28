import faiss
from tqdm import tqdm
from pathlib import Path
import json
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
import copy
import string
from string import punctuation
import os
from os.path import join
import pandas as pd
import json
import argparse
import random

random.seed(1359)

cwd = os.getcwd()
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
beir_dir = join(raw_dir, "beir")
weak_dir = join(data_dir, "weak")
xuyang_data_dir = join(cwd, "xuyang", "data")

def read_json(json_path):
    data = []
    for line in open(json_path, 'r'):
        data.append(json.loads(line))
    return data

def read_weak_json(json_path):
    # read weak query to dict for easily remove unqualified queries
    data = {}
    for line in open(json_path, 'r'):
        weak_q = json.loads(line)
        data[weak_q["_id"]] = weak_q
    return data

def read_text(text_path):
    with open(text_path, "r") as f:
        data = f.readlines()
    return data

def check_data(data_path):
    corpus_path = join(data_path, "corpus.jsonl")
    queries_path = join(data_path, "queries.jsonl")
    train_path = join(data_path, "qrels", "train.tsv")
    dev_path = join(data_path, "qrels", "dev.tsv")
    test_path = join(data_path, "qrels", "test.tsv")
    corpus = read_json(corpus_path)
    queries = read_json(queries_path)
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    return train_data, dev_data, test_data, corpus, queries

def filter_unlabeled_corpus(data_path):
    filtered_corpus = []
    filtered_corpus_path = join(data_path, "corpus_filtered.jsonl")
    if os.path.exists(filtered_corpus_path):
        os.remove(filtered_corpus_path)
        print(f"old {filtered_corpus_path} is deleted ...")
    corpus_path = join(data_path, "corpus.jsonl")
    train_path = join(data_path, "qrels", "train.tsv")
    dev_path = join(data_path, "qrels", "dev.tsv")
    test_path = join(data_path, "qrels", "test.tsv")
    # _id, title, text, metadata
    corpus = read_json(corpus_path)
    labeled_corpus_set = set()
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    df = pd.concat([train_data, dev_data, test_data])
    # df query-id, corpus-id, score
    for _, row in df.iterrows():
        labeled_corpus_set.add(str(row["corpus-id"]))
    for doc in tqdm(corpus):
        if doc["_id"] in labeled_corpus_set:
            continue
        else:
            filtered_corpus.append(doc)
    random.shuffle(filtered_corpus)
    print(f"writing {filtered_corpus_path} ...")
    with open(filtered_corpus_path, "w+") as f:
        for doc in filtered_corpus:
            json.dump(doc, f)
            f.write("\n")

def sample_corpus(dataset_name, ratio: int = 20, train_num: int = 500):
    """sample positive: negative = ration

    Args:
        folder_path (_type_): _description_
    """
    reduced_corpus, sample_negative_corpus = [], []
    filtered_corpus_path = join(beir_dir, dataset_name, "corpus_filtered.jsonl")
    reduced_corpus_path = join(beir_dir, dataset_name, f"corpus_reduced_ratio_{ratio}.jsonl")
    sampled_corpus_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", f"corpus_filtered_5000_id.tsv")
    if os.path.exists(reduced_corpus_path):
        os.remove(reduced_corpus_path)
        print(f"old {reduced_corpus_path} is deleted ...")
    filtered_corpus = read_json(filtered_corpus_path)
    corpus_path = join(beir_dir, dataset_name, "corpus.jsonl")
    train_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", f"prompt_tuning_{train_num}.tsv")
    dev_path = join(beir_dir, dataset_name, "qrels", "dev.tsv")
    test_path = join(beir_dir, dataset_name, "qrels", "test.tsv")
    # _id, title, text, metadata
    corpus = read_json(corpus_path)
    labeled_corpus_set = set()
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    sampled_data = pd.read_csv(sampled_corpus_path, sep='\t')
    df = pd.concat([train_data, dev_data, test_data])
    # df query-id, corpus-id, score
    for _, row in df.iterrows():
        labeled_corpus_set.add(str(row["corpus-id"]))
    for _, row in sampled_data.iterrows():
        labeled_corpus_set.add(str(row["_id"]))
    # sample negative from filtered_corpus-5000
    filtered_corpus_remove_5000 = []
    for filter_doc in tqdm(filtered_corpus):
        if filter_doc["_id"] in labeled_corpus_set:
            continue
        else:
            filtered_corpus_remove_5000.append(filter_doc)
    for doc in tqdm(corpus):
        if doc["_id"] in labeled_corpus_set:
            reduced_corpus.append(doc) 
    sample_num = len(labeled_corpus_set) * ratio
    if sample_num > len(corpus):
        sample_num = len(corpus)
        print(f"{dataset_name} samples all the corpus")
        with open(reduced_corpus_path, "w+") as f:
            for reduce_doc in corpus:
                json.dump(reduce_doc, f)
                f.write("\n")
        return
    random.seed(3490856385)
    sample_negative_corpus = random.sample(filtered_corpus_remove_5000, sample_num)
    assert len(sample_negative_corpus) == sample_num
    reduced_corpus += sample_negative_corpus
    random.shuffle(reduced_corpus)
    print(f"writing {len(reduced_corpus)} documents to {reduced_corpus_path} ...")
    with open(reduced_corpus_path, "w+") as f:
        for reduce_doc in reduced_corpus:
            json.dump(reduce_doc, f)
            f.write("\n")

def sample_corpus_v2(dataset_name, ratio: int = 20, train_num: int = 50, weak_num: str = "100k"):
    """sample positive: negative = ration

    Args:
        folder_path (_type_): _description_
    """
    reduced_corpus, sample_negative_corpus = [], []
    filtered_corpus_path = join(beir_dir, dataset_name, "corpus_filtered.jsonl")
    reduced_corpus_path = join(beir_dir, dataset_name, f"corpus_{weak_num}_reduced_ratio_{ratio}.jsonl")
    sampled_corpus_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", weak_num, f"corpus_filtered_{weak_num}_id.tsv")
    if os.path.exists(reduced_corpus_path):
        os.remove(reduced_corpus_path)
        print(f"old {reduced_corpus_path} is deleted ...")
    filtered_corpus = read_json(filtered_corpus_path)
    corpus_path = join(beir_dir, dataset_name, "corpus.jsonl")
    train_path = join(xuyang_data_dir, f"{dataset_name}_{train_num}", f"prompt_tuning_{train_num}.tsv")
    dev_path = join(beir_dir, dataset_name, "qrels", "dev.tsv")
    test_path = join(beir_dir, dataset_name, "qrels", "test.tsv")
    # _id, title, text, metadata
    corpus = read_json(corpus_path)
    labeled_corpus_set = set()
    train_data = pd.read_csv(train_path, sep='\t')
    dev_data = pd.read_csv(dev_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    sampled_data = pd.read_csv(sampled_corpus_path, sep='\t')
    df = pd.concat([train_data, dev_data, test_data])
    # df query-id, corpus-id, score
    for _, row in df.iterrows():
        labeled_corpus_set.add(str(row["corpus-id"]))
    for _, row in sampled_data.iterrows():
        labeled_corpus_set.add(str(row["_id"]))
    # sample negative from filtered_corpus-weak
    filtered_corpus_remove_weak = []
    for filter_doc in tqdm(filtered_corpus):
        if filter_doc["_id"] in labeled_corpus_set:
            continue
        else:
            filtered_corpus_remove_weak.append(filter_doc)
    for doc in tqdm(corpus):
        if doc["_id"] in labeled_corpus_set:
            reduced_corpus.append(doc) 
    sample_num = len(labeled_corpus_set) * ratio
    if sample_num > len(corpus):
        sample_num = len(corpus)
        print(f"{dataset_name} samples all the corpus")
        with open(reduced_corpus_path, "w+") as f:
            for reduce_doc in corpus:
                json.dump(reduce_doc, f)
                f.write("\n")
        return
    random.seed(3490856385)
    sample_negative_corpus = random.sample(filtered_corpus_remove_weak, sample_num)
    assert len(sample_negative_corpus) == sample_num
    reduced_corpus += sample_negative_corpus
    random.shuffle(reduced_corpus)
    print(f"writing {len(reduced_corpus)} documents to {reduced_corpus_path} ...")
    with open(reduced_corpus_path, "w+") as f:
        for reduce_doc in reduced_corpus:
            json.dump(reduce_doc, f)
            f.write("\n")

def load_dl(dir):
    """
    load dl2019 dl2020 queries and qrels
    """
    queries = {}
    year = dir.split("_")[-1]
    with open(join(dir, f"queries_{year}", "raw.tsv")) as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query
    qrels_binary = read_json(join(dir, "qrel_binary.json"))[0]
    qrels = read_json(join(dir, "qrel.json"))[0]
    return queries, qrels, qrels_binary

def merge_queries(queries, queries_19, queries_20):
    """
    merge ms's queries with the queries of DL2019 and DL2020 for quick retrieval
    """
    ms_queries = copy.deepcopy(queries)
    for qid, query in queries_19.items():
        ms_queries["tr19ec"+qid] = query
    for qid, query in queries_20.items():
        ms_queries["tr20ec"+qid] = query
    return ms_queries

def extract_results(ms_results):
    """
    extract the results of MS, DL2019 and DL2020 
    """
    results, results_19, results_20 = {}, {}, {}
    for qid, res in ms_results.items():
        if "tr19ec" in qid:
            results_19[qid[6:]] = res
        elif "tr20ec" in qid:
            results_20[qid[6:]] = res
        elif "trec2019" in qid:
            results_19[qid[8:]] = res
        elif "trec2020" in qid:
            results_20[qid[8:]] = res
        else:
            results[qid] = res
    return results, results_19, results_20

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
    args = parser.parse_args()
    return args

def main():
    datasets = ["msmarco", "fiqa"]
    weak_nums = ["100k", "5000"]
    for dataset_name in tqdm(datasets):
        for weak_num in weak_nums:
            folder_path = join(beir_dir, dataset_name)
            filter_unlabeled_corpus(folder_path)
            sample_corpus_v2(dataset_name, weak_num=weak_num)

if __name__ == "__main__":
    main()

