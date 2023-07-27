from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
import random
logger = logging.getLogger(__name__)

class WeakDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", qrels_folder: str = "qrels", qrels_file: str = "", weak_query_file: str = "", weak_qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.weak_queries = {}
        self.weak_qrels = {}
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.weak_query_file = os.path.join(data_folder, query_file) if data_folder else weak_query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file
        self.weak_qrels_file = weak_qrels_file
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load_weak_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.weak_query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        self.check(fIn=self.weak_qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.weak_queries):
            
            self._load_weak_queries()

        if os.path.exists(self.weak_qrels_file) and os.path.exists(self.qrels_file):
            self._load_weak_qrels()
            self.weak_queries = {qid: self.weak_queries[qid] for qid in self.weak_qrels}
            logger.info("Loaded %d Golden+Weak Queries.", len(self.weak_queries))
            logger.info("Golden+Weak Query Example: %s", list(self.weak_queries.values())[0])

        return self.corpus, self.weak_queries, self.weak_qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _read_json(self, json_path):
        data = []
        for line in open(json_path, 'r'):
            data.append(json.loads(line))
        return data

    def _load_corpus(self):
        # data = self._read_json(self.corpus_file)
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
    
    def _load_weak_queries(self):
        logger.info("Loading Golden Queries...")
        with open(self.query_file, encoding='utf8') as fIn:
            for line in tqdm(fIn):
                line = json.loads(line)
                self.weak_queries[line.get("_id")] = line.get("text")
        logger.info("Loading Weak Queries...")
        with open(self.weak_query_file, encoding='utf8') as fIn:
            for line in tqdm(fIn):
                line = json.loads(line)
                # for weak query, cut at most 50 tokens
                text = line.get("text")
                text = " ".join(text.split()[:50])
                self.weak_queries[line.get("_id")] = text
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

    def _load_weak_qrels(self):
        logger.info("Loading Golden Qrels...")
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.weak_qrels:
                self.weak_qrels[query_id] = {corpus_id: score}
            else:
                self.weak_qrels[query_id][corpus_id] = score

        logger.info("Loading Weak Qrels...")
        reader = csv.reader(open(self.weak_qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.weak_qrels:
                self.weak_qrels[query_id] = {corpus_id: score}
            else:
                self.weak_qrels[query_id][corpus_id] = score