'''
copy from dpr/train/train_sbert.py, for training sbert with weak data
'''
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import DataLoader
import random
import pathlib, os
import logging
import argparse
from os.path import join
import sys
from tqdm import tqdm
from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
####
cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))
    sys.path.append(join(cwd, "xuyang"))
from weak_data_loader import WeakDataLoader
from zhiyuan.utils import seed_everything
data_dir = join(cwd, "zhiyuan", "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(cwd, "xuyang", "data")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--num_epochs', required=False, default=20, type=int)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--product', required=False, default="cosine", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
model_name = "bert-base-uncased" 
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
seed_everything(42)
# First, we define the transformer model we want to fine-tune
model_name = "bert-base-uncased"
if args.dataset_name == "msmarco":
    train_batch_size = 96
else:
    train_batch_size = 32
num_epochs = 20
# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4
# We set num_labels=1, which predicts a continuous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512)
# Prepare training samples
corpus_keys = list(corpus.keys())
train_samples = []
for qid, pos_passage_dict in tqdm(qrels.items()):
    query = queries[qid]
    neg_cnt = 0
    for pos_pid in pos_passage_dict.keys():
        train_samples.append(InputExample(texts=[query, corpus[pos_pid]["text"]], label=1))
    while neg_cnt < pos_neg_ration*len(pos_passage_dict):
        neg_pid = random.choice(corpus_keys)
        if neg_pid not in pos_passage_dict:
            train_samples.append(InputExample(texts=[query, corpus[neg_pid]["text"]], label=0))
            neg_cnt += 1
# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
dev_pos_neg_ration = 100
dev_corpus_keys = list(dev_corpus.keys())
dev_samples = {}
for dev_qid, dev_pos_passage_dict in tqdm(dev_qrels.items()):
    dev_query = dev_queries[dev_qid]
    dev_samples[dev_qid] = {"query": dev_query, "positive": set(), "negative": set()}
    dev_neg_cnt = 0
    for dev_pos_pid in dev_pos_passage_dict.keys():
        dev_samples[dev_qid]["positive"].add(dev_corpus[dev_pos_pid]["text"])
    while dev_neg_cnt < dev_pos_neg_ration*len(dev_pos_passage_dict):
        dev_neg_pid = random.choice(dev_corpus_keys)
        if dev_neg_pid not in dev_pos_passage_dict:
            dev_samples[dev_qid]["negative"].add(dev_corpus[dev_neg_pid]["text"])
            dev_neg_cnt += 1
# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

# Configure the training
warmup_steps = int(len(train_samples) * num_epochs / train_batch_size * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True,
)
# Save latest model
# model.save(model_save_path)