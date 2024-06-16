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
# from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from weak_data_loader import WeakDataLoader
from tasb_data_loader import TASBalancedDatasetLoader
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
model_name = "tasb"
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
    query_cluster_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"tasb_q_{args.train_num}_clustring.txt")
    pairs_with_teacher_scores = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"tasb_{args.train_num}_teacher_scores.txt")
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
    query_cluster_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"tasb_q_{args.exp_name}_clustring.txt")
    pairs_with_teacher_scores = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"tasb_{args.exp_name}_teacher_scores.txt")
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

# Configuration
# query_model_name = "distilbert/distilbert-base-uncased"
# doc_model_name = "distilbert/distilbert-base-uncased"
query_model_name = "google-bert/bert-base-uncased"
doc_model_name = "google-bert/bert-base-uncased"
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
tasb_model = beir_models.DPR((query_model_name,doc_model_name), share=True)
colbert_model = Checkpoint(join(data_dir, "checkpoints/colbertv2.0"), colbert_config=ColBERTConfig(doc_maxlen=220))
# Combine parameters from both models
combined_parameters = list(tasb_model.q_model.parameters())

# Initialize one optimizer for both models
combined_optimizer = AdamW(combined_parameters, lr=2e-5)

# Prepare training samples (assuming retriever.load_train() is implemented correctly)
batch_size = 40
data_process = TrainRetriever(model=None)
train_samples = data_process.load_train(corpus, queries, qrels)
steps_per_epoch = math.ceil(len(train_samples) / batch_size)
total_steps = steps_per_epoch*num_epochs
train_dataloader = TASBalancedDatasetLoader(corpus, queries, pairs_with_teacher_scores,query_cluster_file, batch_size=batch_size)

def dot_product_similarity(query_embeddings, doc_embeddings):
    # Calculate dot product similarity between query and document embeddings
    # query_embeddings: [batch_size, embedding_size]
    # doc_embeddings: [batch_size, embedding_size]
    # Output: [batch_size, batch_size]
    return torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))

def margin_mse_loss(student_pos, student_neg, teacher_pos, teacher_neg):
    """
    Calculate the Margin-MSE loss for a single pair of positive and negative samples.

    :param student_pos: Scores from the student model for positive samples
    :param student_neg: Scores from the student model for negative samples
    :param teacher_pos: Scores from the teacher model for positive samples
    :param teacher_neg: Scores from the teacher model for negative samples
    :return: The Margin-MSE loss
    """
    # Margin-MSE calculation
    student_diff = student_pos - student_neg
    teacher_diff = teacher_pos - teacher_neg
    loss = (student_diff - teacher_diff) ** 2
    return loss.mean()

def train_loss(query_embeddings, pos_doc_embeddings, neg_doc_embeddings, cat_pos_scores, cat_neg_scores, colbert_pos_scores, colbert_neg_scores, alpha=0.5):
    batch_size = query_embeddings.shape[0]
    pos_student_scores = dot_product_similarity(query_embeddings, pos_doc_embeddings)
    neg_student_scores = dot_product_similarity(query_embeddings, neg_doc_embeddings)
    cat_loss = margin_mse_loss(pos_student_scores.diag(), neg_student_scores.diag(), cat_pos_scores, cat_neg_scores)
    colbert_loss = 0
    
    for i in range(batch_size):
        # Get the student's scores for the positive and all negatives for the i-th query
        student_pos_score = pos_student_scores[i, i]  # Score with the positive document
        student_neg_scores = torch.cat([pos_student_scores[i, :i], pos_student_scores[i, i+1:], neg_student_scores[i]])

        # Get the teacher's scores for the positive and all negatives for the i-th query
        teacher_pos_score = colbert_pos_scores[i, i]  # Score with the positive document
        teacher_neg_scores = torch.cat([colbert_pos_scores[i, :i], colbert_pos_scores[i, i+1:], colbert_neg_scores[i]])

        # Calculate the marginal MSE loss for the i-th query
        loss = margin_mse_loss(student_pos_score, student_neg_scores, teacher_pos_score, teacher_neg_scores)
        colbert_loss += loss
    colbert_loss /= batch_size
    total_loss = cat_loss + colbert_loss * alpha
    return total_loss


def get_colbert_score(Q, D, D_mask):
    assert len(Q) == len(D)
    scores_matrix = torch.zeros(len(Q), len(D))
    for i, q in enumerate(Q):
        q = q.unsqueeze(0)
        scores = colbert_score(q, D, D_mask)
        scores_matrix[i] = scores
    return scores_matrix

tokenizer = tasb_model.q_tokenizer
# Evaluation and Saving Best Model
best_metric = float("-inf")
for epoch in range(num_epochs):
    tasb_model.q_model.train()
    tasb_model.ctx_model.train()
    
    for step, (queries, pos_docs, neg_docs, cat_pos_scores, cat_neg_scores) in tqdm(enumerate(train_dataloader), total=steps_per_epoch-1):
        cat_pos_scores = cat_pos_scores.cuda()
        cat_neg_scores = cat_neg_scores.cuda()
        query_features = tokenizer(queries, truncation=True, padding=True, return_tensors='pt')
        pos_doc_features = tokenizer(pos_docs, truncation=True, padding=True, return_tensors='pt', max_length=max_seq_length)
        neg_doc_features = tokenizer(neg_docs, truncation=True, padding=True, return_tensors='pt', max_length=max_seq_length)

        Q = colbert_model.queryFromText(queries)
        # Q = Q.unsqueeze(0)
        pos_D, pos_D_mask = colbert_model.docFromText(pos_docs, keep_dims='return_mask')
        pos_D_mask = pos_D_mask.squeeze(-1)
        # pos_D, pos_D_mask = colbert_model.docFromText(pos_docs)
        # pos_D_mask = torch.ones(pos_D.shape[:2], dtype=int)
        colbert_pos_scores = get_colbert_score(Q, pos_D, pos_D_mask)

        neg_D, neg_D_mask = colbert_model.docFromText(neg_docs, keep_dims='return_mask')
        neg_D_mask = neg_D_mask.squeeze(-1)
        # neg_D, neg_D_mask = colbert_model.docFromText(neg_docs)
        # neg_D_mask = torch.ones(neg_D.shape[:2], dtype=int)
        colbert_neg_scores = get_colbert_score(Q, neg_D, neg_D_mask)
        
        colbert_pos_scores = colbert_pos_scores.cuda()
        colbert_neg_scores = colbert_neg_scores.cuda()
        query_embeddings = tasb_model.q_model(query_features['input_ids'].cuda(), attention_mask=query_features['attention_mask'].cuda()).last_hidden_state[:,0,:]
        pos_doc_embeddings = tasb_model.ctx_model(pos_doc_features['input_ids'].cuda(), attention_mask=pos_doc_features['attention_mask'].cuda()).last_hidden_state[:,0,:]
        neg_doc_embeddings = tasb_model.ctx_model(neg_doc_features['input_ids'].cuda(), attention_mask=neg_doc_features['attention_mask'].cuda()).last_hidden_state[:,0,:]

        # Assuming the loss function can handle separate embeddings
        loss_value = train_loss(query_embeddings, pos_doc_embeddings, neg_doc_embeddings, cat_pos_scores, cat_neg_scores, colbert_pos_scores, colbert_neg_scores, alpha=0.75)

        loss_value.backward()
        combined_optimizer.step()
        combined_optimizer.zero_grad()
        # print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.item()}")
        if step == steps_per_epoch-1:
            break
    
    # Evaluation after each epoch
    tasb_model.q_model.eval()
    tasb_model.ctx_model.eval()
    eval_model = DRES(beir_models.DPR((query_model_name, doc_model_name), query_model=tasb_model.q_model, doc_model=tasb_model.ctx_model), batch_size=256, corpus_chunk_size=30000)
    eval_retriever = EvaluateRetrieval(eval_model, k_values=[10], score_function="dot")
    dev_results = eval_retriever.retrieve(dev_corpus, dev_queries)
    ndcg, map, recall, _, score_per_query= eval_retriever.evaluate(dev_qrels, dev_results, [100])
    # mrr, mrr_score = eval_retriever.evaluate_custom(dev_qrels, dev_results, [10], metric="mrr")
    # mrr = mrr["MRR@10"]
    metric = recall["Recall@100"]
    if metric > best_metric:
        best_metric = metric
        torch.save(tasb_model.q_model.state_dict(), f"{model_save_path}/query_model_state_dict.pt")
        torch.save(tasb_model.ctx_model.state_dict(), f"{model_save_path}/doc_model_state_dict.pt")
        last_improvement = epoch
        print(f"New best Recall@100: {metric}, models saved.")
    # Early stopping logic based on patience
    if epoch - last_improvement > 10:
        print("No improvement observed for 10 epochs, stopping training.")
        break