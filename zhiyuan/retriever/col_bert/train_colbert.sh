#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -g 0,1 -d msmarco -e noaug -p 2000 -c msmarco.psg.l2/checkpoints/colbert-300000.dnn"
   echo -e "\t-g: cudaNum"
   echo -e "\t-d: datasetname, valid datasetnames are: msmarco, hotpotqa, and fiqa"
   echo -e "\t-e: expname, valid expnames are: noaug, inpairs, and sptadr"
   echo -e "\t-m: max_steps"
   echo -e "\t-s: save checkpoint every s steps"
   echo -e "\t-b: batch size"
   exit 1 # Exit script after printing help
}

while getopts "g:d:e:m:s:b:" opt
do
   case "$opt" in
      g ) cudaNum="$OPTARG" ;;
      d ) datasetname="$OPTARG" ;;
      e ) expname="$OPTARG" ;;
      m ) max_steps="$OPTARG" ;;
      s ) save_checkpoint_steps="$OPTARG" ;;
      b ) batchsize="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$cudaNum" ] || [ -z "$datasetname" ] || [ -z "$expname" ] || [ -z "$max_steps" ] || [ -z "$save_checkpoint_steps" ] || [ -z "$batchsize" ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$cudaNum"
echo "$datasetname"
echo "$expname"
# Config path
export cwd="$(pwd)"
export colbert_dir="${cwd}/zhiyuan/retriever/col_bert"
export data_dir="${colbert_dir}/data/datasets/${datasetname}/${expname}/train"
export raw_data_dir="${cwd}/zhiyuan/datasets/raw/beir/${datasetname}"
export model_dir="${colbert_dir}/data/models/${datasetname}/${expname}"

# Mention Any BEIR dataset here (which has been preprocessed)
export dataset="$datasetname"

# Path where preprocessed collection and queries are present
export COLLECTION="${data_dir}/collection.tsv"
export QUERIES="${data_dir}/queries.tsv"
export TRIPLES="${data_dir}/triples.jsonl"
echo $COLLECTION
echo $QUERIES
echo $TRIPLES
echo $model_dir
echo $expname

# Some Index Name to store the faiss index
export INDEX_NAME="${datasetname}"

# Setting LD_LIBRARY_PATH for CUDA (Incase required!)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ukp/thakur/cuda/lib64/
export TOKENIZERS_PARALLELISM=true
############################################################################
# 1. Train #
############################################################################ 

CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m torch.distributed.launch \
    --nproc_per_node=4 -m zhiyuan.retriever.col_bert.colbert.train \
    --amp \
    --maxsteps ${max_steps} \
    --save_checkpoint_steps ${save_checkpoint_steps}\
    --doc_maxlen 350 \
    --mask-punctuation \
    --bsize ${batchsize}\
    --accum 1 \
    --triples $TRIPLES \
    --collection $COLLECTION \
    --queries $QUERIES \
    --root $model_dir \
    --experiment ${expname} \
    --similarity l2\
    --run "${dataset}.${expname}.l2"

############################################################################
# 2. Train: Single GPU #
############################################################################ 
# python -m zhiyuan.retriever.col_bert.colbert.train \
#     --amp \
#     --maxsteps 315 \
#     --save_checkpoint_steps 15\
#     --doc_maxlen 350 \
#     --mask-punctuation \
#     --bsize 32\
#     --accum 1 \
#     --triples $TRIPLES \
#     --collection $COLLECTION \
#     --queries $QUERIES \
#     --root $model_dir \
#     --experiment ${expname} \
#     --similarity l2\
#     --run "${dataset}.${expname}.l2"