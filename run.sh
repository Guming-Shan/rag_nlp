#!/bin/bash

set -e

DATASET="/home/ubuntu/rag_nlp/pipeline/dataset/financebench_rag.json"
CORPUS="/home/ubuntu/rag_nlp/pipeline/dataset/financebench_corpus.json"

MODELS=(
"/home/ubuntu/rag_nlp/pipeline/models/Qwen/Qwen2___5-7B-Instruct"
"/home/ubuntu/rag_nlp/pipeline/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
)

METHODS=("rag" "llm" "selfrag")
TOPKS=(3 5)

DEBUG_SIZE=40

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for topk in "${TOPKS[@]}"; do

      echo "=================================================="
      echo "Running: method=$method | topk=$topk | model=$model"
      echo "=================================================="

      python main.py \
        --method $method \
        --batch \
        --topk $topk \
        --dataset_path $DATASET \
        --corpus_path $CORPUS \
        --debug \
        --debug_size $DEBUG_SIZE \
        --model_path "$model"

    done
  done
done