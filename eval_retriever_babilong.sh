#!/usr/bin/env bash
# Usage: ./eval_retriever_babilong.sh <pretrained_path> <gpu_id> <seed>
# Example : ./eval_retriever_babilong.sh runs/Jul26_02-56-05_PQN_qa3_three-supporting-facts/ 0 42

set -e  # stop when first error occurs

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <pretrained_path> <gpu_id> <seed>"
  exit 1
fi

PRETRAINED_PATH="$1"
GPU_ID="$2"
SEED="$3"
                   #1k, 4k, 32k, 128k, 1kk    10kk
NUM_SENTENCES_LIST=(50 160 1200 4600 40000) # 400000)

PYTHON="python3" #"$HOME/.mlspace/envs/msr/bin/python3"
SCRIPT="eval_and_log_pqn.py"

for N in "${NUM_SENTENCES_LIST[@]}"; do
  echo "=== Запуск с envs.num_sentences=${N} ==="
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT" \
    pretrained_path="$PRETRAINED_PATH" \
    envs.num_sentences="$N" \
    num_samples=-1 \
    seed="$SEED"
done
