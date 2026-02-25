<div align="center">

# Q-RAG: Long Context Multi‑Step Retrieval via Value‑Based Embedder Training

**🏆 Accepted at ICLR 2026 (Oral)**

[![Paper](https://img.shields.io/badge/Paper-ICLR_2026-blue?style=for-the-badge)](https://openreview.net/forum?id=MS9nWFY7LG)
[![arXiv](https://img.shields.io/badge/arXiv-2511.07328-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2511.07328)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

</div>

**Q-RAG** is a resource-efficient method for **multi-step retrieval** trained with reinforcement learning directly in the latent space of text-chunk embeddings. Instead of expensive LLM fine-tuning, Q-RAG trains only a lightweight embedder agent using value-based RL (temporal difference learning), keeping the LLM frozen. This repository provides the full training and evaluation code to reproduce the results from the paper.

Q-RAG achieves **state-of-the-art results** on long-context benchmarks (**BabiLong**, **RULER**) for contexts up to **10M tokens** and competitive performance on open-domain multi-hop QA (**HotpotQA**, **Musique**) — all trained on a **single A100 GPU**.

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📄 Paper (ICLR 2026) | [OpenReview](https://openreview.net/forum?id=MS9nWFY7LG) |
| 📝 arXiv preprint | [arxiv.org/abs/2511.07328](https://arxiv.org/abs/2511.07328) |
| 📑 PDF | [arxiv.org/pdf/2511.07328](https://arxiv.org/pdf/2511.07328) |
| 💾 Data | [Google Drive](https://drive.google.com/drive/folders/1UUIx-6vEBF9Mij81iVgPul86aXhdyxhG) |

---

### Results
#### RULER benchmark (Table 1)
Q-RAG achieves near-perfect retrieval on all NIAH subtasks, trained on 4K-length documents and generalizing up to 1M tokens:
| Context length | NIAH S1 | NIAH S2 | NIAH S3 | NIAH MK1 | NIAH MV | NIAH MQ | NIAH Avg | Multi-hop QA |
|----------------|---------|---------|---------|---------|---------|---------|----------|--------------|
| 4K  | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 67 |
| 16K | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 67 |
| 32K | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 67 |
| 128K | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 62 |
| 1M  | 100 | 100 | 100 | 100 | 100 | 100 | **99.7** | 57 |
#### Open-domain QA (Table 2)
Results on HotpotQA (in-domain) and Musique (out-of-distribution). QwQ-32B was used as the reader LLM for Q-RAG and Beam Retriever:
| Method | HotpotQA Answer F1 | HotpotQA Fact F1 | Musique Answer F1 | Musique Fact F1 |
|--------|--------------------|--------------------|---------------------|-------------------|
| **Q-RAG** (ours) | **0.76** | **0.95** | **0.52** | **0.71** |
#### BabiLong benchmark
Q-RAG achieves the highest average performance across 5 tasks (QA1–QA5) at context lengths from 1M to 10M tokens, outperforming Titans, Atlas, ARMT, RMT, and proprietary LLMs. On the hardest subtask **QA3** (3-hop temporal reasoning), Q-RAG shows virtually **no degradation** as context grows to 10M tokens, while all baselines degrade significantly.

<p align="center">
  <img src="images/babilong_avg_ans_v2.png" width="48%"/>
  <img src="images/babilong_qa3_ans.png" width="48%"/>
</p>

> **Reproducibility note:** Results may vary slightly across seeds. All training was performed on a single A100-80GB GPU within 12 hours per model.

---

## 🛠 Installation

### Requirements

- Python 3.12
- CUDA-compatible GPU (80 GB A100 recommended for full reproduction)
- Linux recommended (tested on Ubuntu with CUDA)

### Setup

```bash
# Create conda environment
conda create -n qrag python=3.12 -y
conda activate qrag

# Install dependencies
python -m pip install -U pip wheel
pip install vllm  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets
```

### Smoke test

```bash
python -c "from rl.agents.pqn import PQNAgent; print('✅ Q-RAG installed successfully')"
```

---

## 🔬 Reproducibility

### 1. Data Preparation

Download all datasets (HotpotQA, Musique, BabiLong) from [Google Drive](https://drive.google.com/drive/folders/1UUIx-6vEBF9Mij81iVgPul86aXhdyxhG).

Place the downloaded data so that the environment configs can find them. Default paths are set in the corresponding config files under `configs/envs/`.

### 2. Training

Training is launched via `train_q_rag.py`. All hyperparameters are managed by [Hydra](https://hydra.cc/) configs in `configs/`.

#### BabiLong

```bash
# Example: QA2 task, 100 sentences, single GPU with 16 GB
python train_q_rag.py \
  envs.task=qa2_two-supporting-facts \
  envs.num_sentences=100 \
  batch_size=16 \
  accumulate_grads=3
```

#### HotpotQA

```bash
python train_q_rag.py \
  envs=hotpotqa \
  max_action_length=140 \
  envs.max_steps=3 \
  batch_size=16 \
  accumulate_grads=2 \
  eval_episodes=100
```

#### HotpotQA + Musique (combined, GTE embedder)

Edit `configs/training.yaml`:
- `algo: pqn_gte`
- `envs: combined`
- `batch_size: 16` (for A100-80GB)

Then run:

```bash
python train_q_rag.py
```

> **Note:** `max_action_length` and `max_action_length_in_memory` may need adjustment depending on the dataset and GPU memory.

#### Training details (from the paper)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.98, ε=10⁻⁶) |
| Learning rate | 1.5 × 10⁻⁵ with linear warmup (1000 steps) + linear decay to 10% |
| Weight decay | 5 × 10⁻⁴ |
| Gradient clipping | ℓ₂ norm ≤ 2.0 |
| Gradient accumulation | 8 steps (effective batch size = 96) |
| γ / α / λ / τ | 0.99 / 0.05 / 0.5 / 0.02 |
| Max action length | 220 tokens |
| Training time | ≤ 12 hours on a single A100-80GB |

**Models per benchmark:**
- BabiLong & RULER → `facebook/contriever`
- HotpotQA & Musique → `intfloat/multilingual-e5-large`

### 3. Evaluation

#### Retriever evaluation

`eval_retriever.py` evaluates a pretrained retriever checkpoint and writes logs to the model's folder as `eval_seed{seed}_ns{num_sentences}.jsonl`.

```bash
# Single context length
python eval_retriever.py \
  pretrained_path=runs/<run_name> \
  envs.num_sentences=1200 \
  num_samples=200

# HotpotQA
python eval_retriever.py \
  pretrained_path=runs/<run_name> \
  num_samples=-1 \
  envs.max_steps=3
```

**Multi-length BabiLong sweep** (1K → 1M tokens):

```bash
./scripts/eval_retriever_babilong.sh runs/<run_name> 0 42
```

#### LLM evaluation (end-to-end)

Evaluate an LLM on the retriever's output logs:

```bash
# Single log file
CUDA_VISIBLE_DEVICES=0 python eval_llm.py \
  retriever_logdir/retriever_logs.jsonl \
  --llm_name "Qwen/Qwen3-4B" \
  --babi_task qa4

# BabiLong multi-length sweep
./scripts/eval_llm_babilong.sh path/to/retriever_logdir "Qwen/Qwen3-4B" "qa4" 0
```

> **Config priority:** `CLI args` > `configs/testing.yaml` > `pretrained_path/config.yaml`

---

## 📂 Repository Structure

```
Q-RAG/
├── train_q_rag.py              # Main training script
├── eval_retriever.py           # Retriever evaluation
├── eval_llm.py                 # End-to-end LLM evaluation (BabiLong)
├── eval_llm_longbench.py       # LLM evaluation on LongBench
├── eval_sbor_q.py              # Q-value evaluation
├── eval_feedback.py            # Feedback model evaluation
├── llm-answering-vllm.py       # LLM answering via vLLM
│
├── configs/                    # Hydra configs
│   ├── training.yaml           # Main training config
│   ├── testing.yaml            # Evaluation config overrides
│   ├── algo/                   # Algorithm configs (pqn, pqn_gte, …)
│   ├── envs/                   # Environment configs (babilong, hotpotqa, combined, …)
│   ├── feedback/               # Feedback model configs
│   └── logger/                 # Logging configs
│
├── rl/                         # Core RL module
│   ├── agents/                 # Agent implementations (PQN, DQN, SAC-D, SARSA)
│   ├── feedback/               # Feedback / reward models
│   ├── q_module.py             # Q-function neural network
│   ├── optim.py                # Optimizer utilities
│   └── langchain_utils.py      # LangChain integration utilities
│
├── envs/                       # Environments
│   ├── text_env.py             # Text retrieval environment
│   ├── parallel_env.py         # Parallelized environment wrapper
│   ├── qa_env.py               # QA environment
│   ├── dataloaders/            # Dataset loaders (BabiLong, HotpotQA, Musique, …)
│   └── utils.py                # Environment utilities
│
├── prompts_and_metrics/        # Prompts and evaluation metrics
│   ├── babilong.py             # BabiLong prompts & metrics
│   ├── general_qa.py           # General QA metrics
│   ├── chunk_filtering.py      # Chunk filtering logic
│   └── answer_metric.py        # Answer quality metric
│
├── scripts/                    # Shell scripts for batch evaluation
│   ├── eval_retriever_babilong.sh
│   ├── eval_llm_babilong.sh
│   └── train_niah.sh
│
├── notebooks/                  # Jupyter notebooks (examples & analysis)
└── legacy_scripts/             # Archived experimental scripts
```

---

## 📝 Citation

If you find Q-RAG useful, please cite our paper:

```bibtex
@inproceedings{sorokin2026qrag,
  title     = {{Q-RAG}: Long Context Multi-Step Retrieval via Value-Based Embedder Training},
  author    = {Sorokin, Artyom and Buzun, Nazar and Anokhin, Alexander and Inozemcev, Oleg and Vedernikov, Egor and Anokhin, Petr and Burtsev, Mikhail and Trushkov, Alexey and Yin, Wenshuai and Burnaev, Evgeny},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

```bibtex
@article{sorokin2025qrag,
  title   = {{Q-RAG}: Long Context Multi-Step Retrieval via Value-Based Embedder Training},
  author  = {Sorokin, Artyom and Buzun, Nazar and Anokhin, Alexander and Inozemcev, Oleg and Vedernikov, Egor and Anokhin, Petr and Burtsev, Mikhail and Trushkov, Alexey and Yin, Wenshuai and Burnaev, Evgeny},
  journal = {arXiv preprint arXiv:2511.07328},
  year    = {2025}
}
```

---

## ⚖️ License & Acknowledgements

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

We thank the developers of the open-source tools and frameworks that made this work possible, including [Hydra](https://hydra.cc/), [vLLM](https://github.com/vllm-project/vllm), [PyTorch](https://pytorch.org/), [Contriever](https://github.com/facebookresearch/contriever), and [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-large). We also thank the creators of the [BabiLong](https://github.com/booydar/babilong), [RULER](https://github.com/hsiehjackson/RULER), [HotpotQA](https://hotpotqa.github.io/), and [Musique](https://github.com/stonybrooknlp/musique) benchmarks.

For bug reports and questions, please open a [GitHub Issue](../../issues).
