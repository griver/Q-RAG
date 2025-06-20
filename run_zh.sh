#!/usr/bin/env bash
#SBATCH --job-name=babilong_rmt
#SBATCH --output=/trinity/home/a.anokhin/stage_2/pqn/multi-step-retrieval-rl-pqn-29_05/sbatch_logs_hotpot/%x@%A.out
#SBATCH --error=/trinity/home/a.anokhin/stage_2/pqn/multi-step-retrieval-rl-pqn-29_05/sbatch_logs_hotpot/%x@%A.err
#SBATCH --time=16:00:00
#SBATCH --partition=ais-gpu
#SBATCH --gpus-per-task=1         # Запрос двух GPU A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=70G

set -ex

# Параметры путей
SINGULARITY_IMAGE=/trinity/home/a.anokhin/test/rmt_project/recurrent-memory-transformer-aaai24/rmt_big_container.sif
PROJECT_DIR=/trinity/home/a.anokhin/babilong_test/recurrent-memory-transformer

# Переменные для Horovod
export CUDA_VISIBLE_DEVICES=0
export NP=1

# Запуск Bash-скрипта внутри контейнера Singularity с поддержкой CUDA
#srun singularity exec --nv \
#    --bind "${PROJECT_DIR}:/mnt/project" \
#    "${SINGULARITY_IMAGE}" \
#    bash -c "-np ${NP} bash /mnt/project/scripts_exp/babilong/finetune_babilong_qa2_rmt_vary_n_seg_iter_tasks_curriculum.sh"

srun singularity exec --nv --bind /trinity/home/a.anokhin/babilong_test/recurrent-memory-transformer/:/mnt/project /trinity/home/a.anokhin/test/rmt_project/recurrent-memory-transformer-aaai24/rmt_big_container.sif bash -c "python3 train_hotpotqa_pqn.py"