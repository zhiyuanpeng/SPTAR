"""
copy from beir DenseRetrievalExactSearch, for each query, search topk docs from given candadidates instead of the whole corpus
"""
from bm25ce_util import cos_sim, dot_score
import logging
import sys
import torch
from typing import Dict, List
from tqdm import tqdm
logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalExactSearchBM25:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = False #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results_topk, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=self.convert_to_tensor)

        for q_id, corpus_ids in tqdm(results_topk.items()):
            q_index = query_ids.index(q_id)
            query_embedding = query_embeddings[q_index]
            query_embedding = query_embedding.view(1, -1)
            # logger.info("Sorting Corpus by document length (Longest first)...")

            # corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            bm25corpus = [corpus[cid] for cid in corpus_ids]

            # logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            # logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                bm25corpus,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embedding, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            #Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[0])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()[0]
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()[0]
            
            for sub_corpus_id, score in zip(cos_scores_top_k_idx, cos_scores_top_k_values):
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != q_id:
                    self.results[q_id][corpus_id] = score
        
        return self.results 
