#!/usr/bin/env bash
# eval_llm_babilong.sh
#Example
#   ./eval_llm_babilong.sh runs/Jul25_18-11-18_PQN_qa1_single-supporting-fact "Qwen/Qwen3-4B" qa1 1

set -euo pipefail   # exit if error

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <LOGDIR_PATH> <LLM> <TASK> <GPU_ID>"
  exit 1
fi

LOGDIR_PATH=$1      # path to logdir
LLM=$2              # name of llm in hf
TASK=$3             # babi task
GPU_ID=$4           # gpu for inference

SEED=42             # фиксируем сид
NS_LIST=(50 160 1200 4600 40000 400000)   # cycle over number of sentences in context

for NS in "${NS_LIST[@]}"; do
  echo "▶️  Run Num sentences=${NS}, Task=${TASK}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" ~/.mlspace/envs/msr/bin/python3 eval_llm.py \
    "${LOGDIR_PATH}/eval_seed${SEED}_ns${NS}.jsonl" \
    --llm_name "${LLM}" \
    --babi_task "${TASK}" \
    --max_token 256 --think
done
