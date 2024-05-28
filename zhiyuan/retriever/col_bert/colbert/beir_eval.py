import argparse
import logging
from os.path import join
import json
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from collections import defaultdict
import pathlib, os, csv, random
import sys

cwd = os.getcwd()
zhiyuan_dir = join(cwd, "zhiyuan")
if zhiyuan_dir not in sys.path:
    sys.path.append(zhiyuan_dir)
from data_process import load_dl, merge_queries, extract_results
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
csv.field_size_limit(sys.maxsize)

def tsv_reader(input_filepath):
    reader = csv.reader(open(input_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(reader):
        yield idx, row

def eval(args):
    # Configure the basic properties of the logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    # Get your logger
    logger = logging.getLogger(__name__)
    # Assuming args.log_dir is defined somewhere in your script and contains the directory path
    log_file_path = join(args.log_dir, "test_log.txt")
    # Create a file handler and add it to your logger
    handler = logging.FileHandler(log_file_path)
    logger.addHandler(handler)
    
    #### Provide the data_dir where nfcorpus has been downloaded and unzipped
    logger.info("log test-before")
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_dir).load(split=args.split)
    logger.info("log test-after")
    if args.dataset == "msmarco":
        queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
        queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))


    inv_map, results = {}, defaultdict(list)

    #### Document mappings (the ranking.tsv, the doc_id is the order of that doc in collection.tsv)
    for idx, row in tsv_reader(args.collection):
        inv_map[str(idx)] = row[0]

    #### Results ####
    for _, row in tsv_reader(args.rankings):
        qid, doc_id, rank = row[0], row[1], int(row[2])
        if qid != inv_map[str(doc_id)]:
            if qid not in results:
                results[qid] = {inv_map[str(doc_id)]: 1 / (rank + 1)}
            else:
                results[qid][inv_map[str(doc_id)]] = 1 / (rank + 1)
    # score_function by default is cos_sim. score_function is only used for retriever.retrieve, for colbert, this is usused. colbert use l2 distance
    retriever = EvaluateRetrieval(k_values=[1,3,5,10,100,300, 500, 1000], score_function="cos_sim")
    tobe_eval = {}
    if args.dataset == "msmarco":
        results, results_19, results_20 = extract_results(results)
        tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
        tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
    tobe_eval[args.dataset] = (qrels, results, "pad")

    for dataset_name in tobe_eval.keys():
        qrels, results, qrels_binary = tobe_eval[dataset_name]
        logger.info("Retriever evaluation for dataset {}".format(dataset_name))
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
            logger.info("\n")
            for k in mrr.keys():
                logger.info("{}: {:.4f}".format(k, mrr[k]))
        for eval in [ndcg, map, recall]:
                logger.info("\n")
                for k in eval.keys():
                    logger.info("{}: {:.4f}".format(k, eval[k]))
        with open(join(args.log_dir, f"{dataset_name}.json"), "w") as f:
            json.dump(score_per_query, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="BEIR Dataset Name, eg. nfcorpus")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to a BEIR repository (incase already downloaded or custom)')
    parser.add_argument('--log_dir', type=str, default=None, help='Path to log the BEIR results')
    parser.add_argument('--collection', type=str, help='Path to the ColBERT collection file')
    parser.add_argument('--rankings', required=True, type=str, help='Path to the ColBERT generated rankings file')
    # parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10])
    args = parser.parse_args()
    eval(args)
if __name__ == "__main__":
    main()