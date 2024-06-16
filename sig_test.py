#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   sig_test.py
@Time    :   2024/05/30 16:22:38
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   None
'''
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
from scipy import stats
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))

# Configure the basic properties of the logger
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# Get your logger
logger = logging.getLogger(__name__)
# Assuming args.log_dir is defined somewhere in your script and contains the directory path
log_file_path = join(cwd, "p_test_log.txt")
# Create a file handler and add it to your logger
handler = logging.FileHandler(log_file_path)
logger.addHandler(handler)

def getPath(retriever_name, dataset_name, method):
    if retriever_name == "DPR":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr/train/output/no_aug/50/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr/train/output/no_aug/50/bert-base-uncased-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr/train/output/p_written_100k_vicuna_prompt_2_filtered_70/50/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr/train/output/p_written_100k_vicuna_prompt_3_filtered_30/50/bert-base-uncased-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/bert-base-uncased-v1-msmarco"
    if retriever_name == "ColBERT":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/col_bert/data/models/fiqa/no_aug/test/colbert-80"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/col_bert/data/models/msmarco/no_aug/test/colbert-40"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/col_bert/data/models/fiqa/p_written_100k_vicuna_prompt_2_filtered_70/test/colbert-1080"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/col_bert/data/models/msmarco/p_written_100k_vicuna_prompt_3_filtered_30/test/colbert-6300"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/col_bert/data/models/fiqa/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/test/colbert-1560"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/col_bert/data/models/msmarco/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/test/colbert-6460"
    if retriever_name == "BM25DPR":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/no_aug/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/no_aug/50/top_1000/bert-base-uncased-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/p_written_100k_vicuna_prompt_2_filtered_70/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/p_written_100k_vicuna_prompt_3_filtered_30/50/top_1000/bert-base-uncased-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25dpr/eval/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/top_1000/bert-base-uncased-v1-msmarco"
    if retriever_name == "BM25CE":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25ce/eval/output/no_aug/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25ce/eval/output/no_aug/50/top_1000/bert-base-uncased-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25ce/eval/output/p_written_100k_vicuna_prompt_2_filtered_70/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25ce/eval/output/p_written_100k_vicuna_prompt_3_filtered_30/50/top_1000/bert-base-uncased-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/bm25ce/eval/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/top_1000/bert-base-uncased-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/bm25ce/eval/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/top_1000/bert-base-uncased-v1-msmarco"
    if retriever_name == "Contriever":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/no_aug/50/facebook/contriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/no_aug/50/facebook/contriever-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/p_written_100k_vicuna_prompt_2_filtered_70/50/facebook/contriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/p_written_100k_vicuna_prompt_3_filtered_30/50/facebook/contriever-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/facebook/contriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_contriever/train/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/facebook/contriever-v1-msmarco"
    if retriever_name == "ReContriever":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/no_aug/50/Yibin-Lei/ReContriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/no_aug/50/Yibin-Lei/ReContriever-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/p_written_100k_vicuna_prompt_2_filtered_70/50/Yibin-Lei/ReContriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/p_written_100k_vicuna_prompt_3_filtered_30/50/Yibin-Lei/ReContriever-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/Yibin-Lei/ReContriever-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_recontriever/train/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/Yibin-Lei/ReContriever-v1-msmarco"
    if retriever_name == "tas-b":
        if method == "w/o aug":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/no_aug/50/tasb-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/no_aug/50/tasb-v1-msmarco"
        if method == "inpars":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/p_written_100k_vicuna_prompt_2_filtered_70/50/tasb-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/p_written_100k_vicuna_prompt_3_filtered_30/50/tasb-v1-msmarco"
        if method == "sptar":
            if dataset_name == "fiqa":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/tasb-v1-fiqa"
            if dataset_name == "msmarco":
                dir = "zhiyuan/retriever/dpr_tasb/train/output/llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30/50/tasb-v1-msmarco"
    if dataset_name == "fiqa":
        return join(cwd, dir, "fiqa.json")
    if dataset_name == "msmarco":
        return join(cwd, dir, "msmarco.json"), join(cwd, dir, "dl2019.json"), join(cwd, dir, "dl2020.json")

def tt_test(file_a, file_b, metric):
    with open(file_a, "r") as f:
        data_a = json.load(f)[metric]
    with open(file_b, "r") as f:
        data_b = json.load(f)[metric]
    # for MRR@1000, we need to add the mrr@1000=0 for missing queries
    if metric == "MRR@10":
        for k in data_a.keys():
            if k not in data_b:
                data_b[k] = 0
        for k in data_b.keys():
            if k not in data_a:
                data_a[k] = 0
    assert len(data_a) == len(data_b)
    # assert list(data_a.keys()) == list(data_b.keys())
    v_a, v_b = [], []
    for k, v in data_a.items():
        v_a.append(v)
        v_b.append(data_b[k])
    rel_t_stat, rel_p_value = stats.ttest_rel(v_a, v_b, alternative='greater')
    # rel_t_stat, rel_p_value = stats.wilcoxon(v_a, v_b, alternative='greater')
    return rel_p_value 
   
def sigTest(retriever_name, dataset_name, method_b):
    logger.info(f"Start to compare {retriever_name} on {dataset_name} with sptar vs {method_b}")
    if dataset_name == "fiqa":
        fiqa_sptar = getPath(retriever_name, dataset_name, "sptar")
        fiqa_second = getPath(retriever_name, dataset_name, method_b)
        for metric in ["MRR@10", "Recall@100"]:
        # for metric in ["Recall@100"]:
            p_fiqa = tt_test(fiqa_sptar, fiqa_second, metric)
            logger.info(f"p_fiqa with metric {metric}, sptar vs {method_b}: {str(p_fiqa)}")
    if dataset_name == "msmarco":
        msmarco_sptar, dl2019_sptar, dl2020_sptar = getPath(retriever_name, dataset_name, "sptar") 
        msmarco_second, dl2019_second, dl2020_second = getPath(retriever_name, dataset_name, method_b)
        for metric in ["MRR@10", "Recall@100"]:
        # for metric in ["Recall@100"]:
            p_msmarco = tt_test(msmarco_sptar, msmarco_second, metric)
            logger.info(f"p_msmarco with metric {metric}, sptar vs {method_b}: {str(p_msmarco)}")
        for metric in ["MAP@1000", "Recall@100", "NDCG@10"]:
            p_dl2019 = tt_test(dl2019_sptar, dl2019_second, metric)
            p_dl2020 = tt_test(dl2020_sptar, dl2020_second, metric)
            logger.info(f"p_dl2019 with metric {metric}, sptar vs {method_b}: {str(p_dl2019)}")
            logger.info(f"p_dl2020 with metric {metric}, sptar vs {method_b}: {str(p_dl2020)}")

def main():
    for retriever_name in ["DPR", "ColBERT", "BM25DPR", "BM25CE", "Contriever", "ReContriever", "tas-b"]:
        for dataset_name in ["fiqa", "msmarco"]:
            for method_b in ["w/o aug", "inpars"]:
                sigTest(retriever_name, dataset_name, method_b)

if __name__ == "__main__":
    main()



