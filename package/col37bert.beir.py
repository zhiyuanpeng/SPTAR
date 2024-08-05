#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   col37bert.beir.pyt
@Time    :   2024/05/17 18:02:52
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   for colbert, when evaluating the checkpoint on test dataset, please replace the following function in col37bert's beir for saving the results per query:
'''
# mrr, mrr_score = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
def mrr(qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    mrr_per_q = defaultdict(dict)
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    mrr_per_q[f"MRR@{k}"][query_id] = 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR, mrr_per_q

# retriever.evaluate(qrels, results, retriever.k_values)
from collections import defaultdict
def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        score_per_query = defaultdict(dict)
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                score_per_query[f"NDCG@{k}"][query_id] = scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                score_per_query[f"MAP@{k}"][query_id] = scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                score_per_query[f"Recall@{k}"][query_id] = scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
                score_per_query[f"P@{k}"][query_id] = scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        # for eval in [ndcg, _map, recall, precision]:
        #     logging.info("\n")
        #     for k in eval.keys():
        #         logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision, score_per_query