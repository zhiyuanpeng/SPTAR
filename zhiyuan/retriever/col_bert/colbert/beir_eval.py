from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, csv, random
import sys
import argparse
import logging
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="BEIR Dataset Name, eg. nfcorpus")
parser.add_argument('--split', type=str, default="test")
parser.add_argument('--data_dir', type=str, default=None, help='Path to a BEIR repository (incase already downloaded or custom)')
parser.add_argument('--log_dir', type=str, default=None, help='Path to log the BEIR results')
parser.add_argument('--collection', type=str, help='Path to the ColBERT collection file')
parser.add_argument('--rankings', required=True, type=str, help='Path to the ColBERT generated rankings file')
parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10])
args = parser.parse_args()
# main(**vars(args))
csv.field_size_limit(sys.maxsize)

def tsv_reader(input_filepath):
    reader = csv.reader(open(input_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(reader):
        yield idx, row

# def main(dataset, split, data_dir, log_dir, collection, rankings, k_values):
#### Just some code to print debug information to stdout
handler = logging.FileHandler(join(args.log_dir, "test_log.txt"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])
#### Provide the data_dir where nfcorpus has been downloaded and unzipped
# if args.data_dir == None:
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
#     out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
#     args.data_dir = util.download_and_unzip(url, out_dir)

#### Provide the data_dir where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=args.data_dir).load(split=args.split)

inv_map, results = {}, {}

#### Document mappings (from original string to position in tsv file ####
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

#### Evaluate your retrieval using NDCG@k, MAP@K ...
evaluator = EvaluateRetrieval()
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, args.k_values)
mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k_values, metric='mrr')
recall_cap = EvaluateRetrieval.evaluate_custom(qrels, results, args.k_values, metric="r_cap")
hole = EvaluateRetrieval.evaluate_custom(qrels, results, args.k_values, metric="hole")

with open(join(args.log_dir, "test_log.txt"), "w+") as f:
    for eval in [ndcg, _map, recall, mrr, precision, recall_cap, hole]:
        f.write("\n")
        for k in eval.keys():
            f.write("{}: {:.4f}".format(k, eval[k]))
            f.write("\n")
#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


# if __name__ == '__main__':


