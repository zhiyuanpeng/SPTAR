from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from os.path import exists, join
import logging
import pathlib, os
cwd = os.getcwd()
data_dir = join(cwd, "zhiyuan", "datasets")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
out_dir = os.path.join(data_dir, "raw", "beir")
os.makedirs(out_dir, exist_ok=True)
datasets = ["msmarco", "fiqa"]
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)
    print(f"Dataset {dataset} download successfully ...")