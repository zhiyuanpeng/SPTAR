from rich.console import Console
import random
import numpy as np
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List
import logging
import torch
import torch.multiprocessing as mp

class TASBalancedDatasetLoader():
    """
    dynamically samples queries from given cluster information for a batch
    """

    def __init__(
        self,
        collection,
        queries,
        pairs_with_teacher_scores: str,
        query_cluster_file: str,
        batch_size: int = 32,
        clusters_per_batch: int = 1,
        pair_balancing_strategy="bins",  # or "random" or "hard-margin"
        random_seed=42,
    ):

        self.queries = queries
        self.collection = collection
        self.pairs_with_teacher_scores = pairs_with_teacher_scores
        self.query_cluster_file = query_cluster_file
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch
        self.read_with_scores = True
        self.pair_balancing_strategy = pair_balancing_strategy
        self.uniform_percentile_sampling = pair_balancing_strategy == "bins"
        self.uniform_percentile_sampling_bins = 10
        self.seed = random_seed

    def __iter__(self):
        
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

        queue: mp.JoinableQueue = ctx.JoinableQueue(2000)
        worker = ctx.Process(
            target=self.data_loader_subprocess, args=(queue,), daemon=True
        )
        worker.start()

        try:
            for batch, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise Exception(f"Error in data loading worker: {e}\n{tb}")
                yield batch
                queue.task_done()
        finally:
            if hasattr(queue, "close"):  # for compat with different Python versions.
                queue.close()  # type: ignore[attr-defined]
            if worker.is_alive():
                worker.terminate()

    def load_data(self):

        console = Console()

        self.pairs_with_teacher_scores_by_qid = defaultdict(list)
        with open(self.pairs_with_teacher_scores, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split()  # pos_score<t>neg_score<t>pos_id<t>neg_id
                # margins.append(float(ls[0])-float(ls[1]))
                # if self.hard_margin_sampling and float(ls[0])-float(ls[1]) > self.hard_margin_sampling_cutoff:
                #    continue
                self.pairs_with_teacher_scores_by_qid[ls[2]].append((ls[3], ls[4].strip(), float(ls[0]), float(ls[1])))

        if self.uniform_percentile_sampling:
            console.log("[TASBalanced] Creating balanced bins")
            pairs_with_teacher_scores_by_qid_binned = defaultdict(list)
            avg_bin_lengths = [[] for _ in range(self.uniform_percentile_sampling_bins)]
            for q_id, pair_list in self.pairs_with_teacher_scores_by_qid.items():
                if len(pair_list) >= 2:
                    margins = np.array([l[2] - l[3] for l in pair_list])
                    indices = np.digitize(margins, np.arange(np.min(margins), np.max(margins), (np.max(margins)-np.min(margins))/self.uniform_percentile_sampling_bins))
                    bins = [[] for _ in range(self.uniform_percentile_sampling_bins)]
                    for i, p in enumerate(pair_list):
                        bins[indices[i]-1].append(p)
                    for i, b in enumerate(bins):
                        avg_bin_lengths[i].append(len(b))
                    pairs_with_teacher_scores_by_qid_binned[q_id] = bins
            #for i, b in enumerate(avg_bin_lengths):
            #    print("bin", i, "avg:", np.mean(b),"num == 0",np.sum(np.array(b) == 0))
            self.pairs_with_teacher_scores_by_qid = pairs_with_teacher_scores_by_qid_binned

        console.log("[TASBalanced] Loading cluster assignments from:",self.query_cluster_file)
        self.query_clusters = []
        all_cluster_ids = []
        with open(self.query_cluster_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.strip().split(",")  # id<\t>id ....
                self.query_clusters.append(ls)
                all_cluster_ids.extend(ls)

        self.query_ids = set(self.pairs_with_teacher_scores_by_qid.keys()).intersection(set(all_cluster_ids))

        # clean clusters, to only have matching ids with pair file
        for i, c in enumerate(self.query_clusters):
            self.query_clusters[i] = list(set(c).intersection(self.query_ids))
        self.query_clusters = [c for c in self.query_clusters if len(c) > 0]

        console.log("[TASBalanced] Done loading! Using ", len(self.query_ids), " queries from ", len(self.query_clusters),"clusters for seed:",self.seed," with pair_balancing_strategy: ",self.pair_balancing_strategy)

    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        try:
            self.load_data()

            query_target_count = int((self.batch_size / self.clusters_per_batch))

            while True:

                queries, pos_docs, neg_docs = [], [], []
                cat_pos_scores, cat_neg_scores = [], []
                while len(queries) < self.batch_size:

                    # get rnd cluster
                    c_idx = random.randint(0, len(self.query_clusters)-1)

                    # take a query sample out of that cluster
                    if query_target_count < len(self.query_clusters[c_idx]):
                        q_ids = random.sample(self.query_clusters[c_idx], query_target_count)
                    else:
                        q_ids = self.query_clusters[c_idx]

                    for q_id in q_ids:
                        query_text = self.queries[q_id]
                        queries.append(query_text)
                        if self.uniform_percentile_sampling:
                            pair = None
                            while pair == None:
                                bin_idx = random.randint(0, len(self.pairs_with_teacher_scores_by_qid[q_id])-1)
                                if len(self.pairs_with_teacher_scores_by_qid[q_id][bin_idx]) > 0:
                                    pair = random.choice(self.pairs_with_teacher_scores_by_qid[q_id][bin_idx])

                        else:
                            pair = random.choice(self.pairs_with_teacher_scores_by_qid[q_id])

                        pos_docs.append(self.collection[pair[0]]['text'])
                        neg_docs.append(self.collection[pair[1]]['text'])
                        cat_pos_scores.append(pair[2])
                        cat_neg_scores.append(pair[3])
                        # main_instances
                        if len(queries) == self.batch_size:
                            break
                main_batch = (queries, pos_docs, neg_docs, torch.tensor(cat_pos_scores), torch.tensor(cat_neg_scores))

                queue.put((main_batch,None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))
        
        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()