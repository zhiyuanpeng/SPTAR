#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -g 0,1 -d msmarco -e noaug -p 2000 -c msmarco.psg.l2/checkpoints/colbert-300000.dnn"
   echo -e "\t-d: datasetname, valid datasetnames are: msmarco, hotpotqa, and fiqa"
   echo -e "\t-e: expname, valid expnames are: noaug, inpairs, and sptadr"
   echo -e "\t-c: checkpoint name"
   exit 1 # Exit script after printing help
}

while getopts "d:e:c:" opt
do
   case "$opt" in
      d ) datasetname="$OPTARG" ;;
      e ) expname="$OPTARG" ;;
      c ) checkpointname="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$datasetname" ] || [ -z "$expname" ] || [ -z "$checkpointname" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$datasetname"
echo "$expname"
echo "$checkpointname"
# Config path
export cwd="$(pwd)"
export colbert_dir="${cwd}/zhiyuan/retriever/col_bert"
export data_dir="${colbert_dir}/data/datasets/${datasetname}/${expname}/dev"
export raw_data_dir="${cwd}/zhiyuan/datasets/raw/beir/${datasetname}"
export model_dir="${colbert_dir}/data/models/${datasetname}/${expname}"

# Mention Any BEIR dataset here (which has been preprocessed)
export dataset="$datasetname"

export CHECKPOINT="${model_dir}/${expname}/train.py/${dataset}.${expname}.l2/checkpoints/colbert-${checkpointname}.dnn"
echo $CHECKPOINT
# Path where preprocessed collection and queries are present
export COLLECTION="${data_dir}/collection.tsv"
export QUERIES="${data_dir}/queries.tsv"
export QRELS="${data_dir}/qrels.jsonl"
echo $COLLECTION
echo $QUERIES
echo $TRIPLES
echo $model_dir
echo $expname

# Setting LD_LIBRARY_PATH for CUDA (Incase required!)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ukp/thakur/cuda/lib64/

############################################################################
# 1. Val #
############################################################################ 

python -m zhiyuan.retriever.col_bert.colbert.test \
    --amp \
    --doc_maxlen 350\
    --mask-punctuation \
    --collection $COLLECTION \
    --queries $QUERIES \
    --qrels $QRELS \
    --checkpoint $CHECKPOINT \
    --root $model_dir \
    --experiment ${expname} \