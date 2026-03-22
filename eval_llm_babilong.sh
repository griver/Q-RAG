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

SEED=42             
NS_LIST=(50 160 1200 4600 40000 400000)   # cycle over number of sentences in context
QVALUE_LIST=(0.5)   # qvalue threshold for chunk filtering

for QVALUE in "${QVALUE_LIST[@]}"; do
  # Create log file with qvalue prefix
  LOG_FILE="${LOGDIR_PATH}/eval_llm_qvalue_${QVALUE}_task_${TASK}.log"
  echo "Starting evaluation with qvalue=${QVALUE}, task=${TASK}, log: ${LOG_FILE}"
  
  for NS in "${NS_LIST[@]}"; do
    echo "Run Num sentences=${NS}, Task=${TASK}, qvalue=${QVALUE}"
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 eval_llm_synthetics.py \
      "${LOGDIR_PATH}/eval_seed${SEED}_ns${NS}_max-steps6.jsonl" \
      --llm_name "${LLM}" \
      --babi_task "${TASK}" \
      --chunk_filter qvalue \
      --stopping_threshold "${QVALUE}" \
      >> "${LOG_FILE}" 2>&1
    echo "Completed NS=${NS} with qvalue=${QVALUE}" | tee -a "${LOG_FILE}"
  done
  
  echo "Completed all NS values for qvalue=${QVALUE}"
done
