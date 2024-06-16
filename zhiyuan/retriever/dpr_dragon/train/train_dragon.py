'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models as beir_models
import pathlib, os
import logging
import argparse
from os.path import join
import math
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from weak_data_loader import WeakDataLoader
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--num_epochs', required=False, default=20, type=int)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--product', required=False, default="cosine", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
model_name = "dragon"
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), "{}-v1-{}".format(model_name, args.dataset_name))
os.makedirs(model_save_path, exist_ok=True)
#### Just some code to print debug information to stdout
fh = logging.FileHandler(join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])
#### /print debug information to stdout
#### Provide the data_path where nfcorpus has been downloaded and unzipped
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

# Configuration
query_model_name = "facebook/dragon-plus-query-encoder"
doc_model_name = "facebook/dragon-plus-context-encoder"
max_seq_length = 300
num_epochs = args.num_epochs

# Define Models for Query and Document
# query_word_embedding_model = models.Transformer(query_model_name, max_seq_length=max_seq_length)
# query_pooling_model = models.Pooling(query_word_embedding_model.get_word_embedding_dimension())
# query_model = SentenceTransformer(modules=[query_word_embedding_model, query_pooling_model]).cuda()
# query_model_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
# doc_word_embedding_model = models.Transformer(doc_model_name, max_seq_length=max_seq_length)
# doc_pooling_model = models.Pooling(doc_word_embedding_model.get_word_embedding_dimension(),
#                                    pooling_mode_cls_token=True,     # Enable using the CLS token
#                                    pooling_mode_mean_tokens=False,  # Disable mean pooling
#                                    pooling_mode_max_tokens=False)   # Disable max pooling
# doc_model = SentenceTransformer(modules=[doc_word_embedding_model, doc_pooling_model]).cuda()
# doc_model_tokenizer = AutoTokenizer.from_pretrained(doc_model_name)
dragon_model = beir_models.DPR((query_model_name,doc_model_name))

# Combine parameters from both models
combined_parameters = list(dragon_model.q_model.parameters()) + list(dragon_model.ctx_model.parameters())

# Initialize one optimizer for both models
combined_optimizer = AdamW(combined_parameters, lr=2e-5)

def collate_fn(batch):
    queries = [sample.texts[0] for sample in batch]
    docs = [sample.texts[1] for sample in batch]
    return {'queries': queries, 'docs': docs}
# Prepare training samples (assuming retriever.load_train() is implemented correctly)
batch_size = 80
data_process = TrainRetriever(model=None)
train_samples = data_process.load_train(corpus, queries, qrels)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

def dot_product_similarity(query_embeddings, doc_embeddings):
    # Calculate dot product similarity between query and document embeddings
    # query_embeddings: [batch_size, embedding_size]
    # doc_embeddings: [batch_size, embedding_size]
    # Output: [batch_size, batch_size]
    return torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))

def train_loss(query_embeddings, doc_embeddings):
    # Compute similarity scores using dot product
    scores = dot_product_similarity(query_embeddings, doc_embeddings)
    
    # Generate labels for each query
    # Assuming each query's positive document is at the same index in doc_embeddings
    labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device)
    
    # Apply cross-entropy loss
    # CrossEntropyLoss expects scores to be [batch_size, num_classes] and labels to be [batch_size]
    # where each label is the index of the true class in the corresponding row of scores
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(scores, labels)
    return loss

# Evaluation and Saving Best Model
best_metric = float("-inf")
for epoch in range(num_epochs):
    dragon_model.q_model.train()
    dragon_model.ctx_model.train()
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        query_features = dragon_model.q_tokenizer(batch['queries'], truncation=True, padding=True, return_tensors='pt')
        doc_features = dragon_model.ctx_tokenizer(batch['docs'], truncation=True, padding=True, return_tensors='pt', max_length=max_seq_length)
        query_embeddings = dragon_model.q_model(query_features['input_ids'].cuda(), attention_mask=query_features['attention_mask'].cuda()).last_hidden_state[:,0,:]
        doc_embeddings = dragon_model.ctx_model(doc_features['input_ids'].cuda(), attention_mask=doc_features['attention_mask'].cuda()).last_hidden_state[:,0,:]
        # Assuming the loss function can handle separate embeddings
        loss_value = train_loss(query_embeddings, doc_embeddings)
        loss_value.backward()
        combined_optimizer.step()
        combined_optimizer.zero_grad()
        # print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.item()}")
        # break
    
    # Evaluation after each epoch
    dragon_model.q_model.eval()
    dragon_model.ctx_model.eval()
    eval_model = DRES(beir_models.DPR((query_model_name, doc_model_name), query_model=dragon_model.q_model, doc_model=dragon_model.ctx_model), batch_size=256, corpus_chunk_size=30000)
    eval_retriever = EvaluateRetrieval(eval_model, k_values=[10], score_function="dot")
    dev_results = eval_retriever.retrieve(dev_corpus, dev_queries)
    ndcg, map, recall, _, score_per_query= eval_retriever.evaluate(dev_qrels, dev_results, [100])
    # mrr, mrr_score = eval_retriever.evaluate_custom(dev_qrels, dev_results, [10], metric="mrr")
    # mrr = mrr["MRR@10"]
    metric = recall["Recall@100"]
    if metric > best_metric:
        best_metric = metric
        torch.save(dragon_model.q_model.state_dict(), f"{model_save_path}/query_model_state_dict.pt")
        torch.save(dragon_model.ctx_model.state_dict(), f"{model_save_path}/doc_model_state_dict.pt")
        last_improvement = epoch
        print(f"New best Recall@100: {metric}, models saved.")
    # Early stopping logic based on patience
    if epoch - last_improvement > 10:
        print("No improvement observed for 10 epochs, stopping training.")
        break