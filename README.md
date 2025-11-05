# Multi-Step Retrieval via Reinforcement Learning 

## Installation

Create and activate a conda environment:

```bash
conda create -n qrag python=3.12 -y
conda activate qrag
```

Upgrade pip and install the Python dependencies:

```bash
python -m pip install -U pip wheel
pip install vllm # pulls in compatible versions of PyTorch, Transformers, Triton, etc.
pip hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets
```


## Training
To train Contriever Embedder with Q-RAG on Babilong see `train_pqn.py`. 
To modify hyperparameters either use yaml configs or CLI.
Use `train_q_rag.py` to train the Contriever embedder with Q-RAG. The script supports the **Babilong**, **HotpotQA** and **Musique** datasets.

### Datasets
All HotPotQA, Musique and BabiLong data can be downloaded from the following link: [Google Drive](https://drive.google.com/drive/folders/1UUIx-6vEBF9Mij81iVgPul86aXhdyxhG).


### Configs
All hyperparameters are set in `configs/`. Useful files include:
* `configs/training.yaml`
* `configs/envs/babilong.yaml`
* `configs/envs/hotpotqa.yaml`
* `configs/algo/pqn.yaml`

### CLI
You can change any config you want by directly passing it into training script:
Example – Babilong (for a single GPU with 16 GB):
```bash
python train_q_rag.py envs.task=qa2_two-supporting-facts envs.num_sentences=100 batch_size=16 accumulate_grads=3
```
Example. HotpotQA:
```bash
python train_q_rag.py envs=hotpotqa max_action_length=140 envs.max_steps=3 batch_size=16 accumulate_grads=2 eval_episodes=100
```
Example. For training Q-RAG with GTE embedder on the combined dataset HotpotQA+Musique, modify `training.yaml` config file:
* `algo: pqn_gte`
* `envs: combined`
* `batch_size: 16`  (for A100-80GB)

`max_action_length` and `max_action_length_in_memory` may also need to be changed.
Then run the training script:
```bash
python3 train_q_rag.py
```

## Testing
`eval_retriever.py` evaluates a pretrained retriever and stores logs in the model's folder. The log filename depends on the evaluation seed and the number of sentences (in case of Babilong):
`eval_seed{seed}_ns{num_sentences}.jsonl` and is written to `pretrained_path`.
For example, with `seed=42` and `envs.num_sentences=160` the log will be `eval_seed42_ns160.jsonl`.

Example – evaluating the retriever:
```bash
python eval_retriever.py pretrained_path=runs/May30_03-44-01_PQN_qa2_two-supporting-facts envs.num_sentences=1200 num_samples=200
```

Example – evaluating the retriever on HotpotQA:
```bash
python eval_retriever.py pretrained_path=runs/Jul18_17-26-55_PQN_hotpotqa  num_samples=-1 envs.max_steps=3
```
Testing only hyperparams are stored in `configs/testing.yaml`. Hyperparameters specified in CLI or `configs/testing.yaml` overwrites values from the config in the pretrained_path. 
Priority between all sources is the following:

`CLI hyperparams > configs/testing.yaml > pretrained_path/config.yaml`, 
where `A > B` means that `A.param1` overwrites `B.param1`.

The `eval_retriever_babilong.sh` script runs `eval_retriever.py` over multiple context lengths (1k-1m tokens) for a Babilong task:
```bash
./eval_retriever_babilong.sh runs/Jul26_02-56-05_PQN_qa3_three-supporting-facts/ 0 42
```

### Evaluating the LLM from retriever logs
To test an LLM on a single log file:
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_llm.py retriever_logdir/retriever_logs.jsonl --llm_name "Qwen/Qwen3-4B" --babi_task qa4
```

To evaluate all Babilong logs for different context length (1k-1m tokens) in a directory:
```bash
./eval_llm_babilong.sh path/to/retriever_logdir "Qwen/Qwen3-4B" "qa4" 0
```

`eval_llm.py`, `eval_retriever_babilong.sh`, and `eval_llm_babilong.sh` are currently tailored for Babilong-specific prompts.


## In Progress

### Feedback models
Reward functions for the environments are selected via the `feedback.type` key,
which maps to entries in `configs/feedback/defaults.yaml`. Override this value to
choose a different feedback model:

```bash
python train_q_rag.py feedback.type=gt          # ground truth feedback
python train_q_rag.py feedback.type=babilong_em # LLM exact‑match feedback
```

### Using AnswerMetricFeedback
The `babilong_em` option relies on `rl.feedback.AnswerMetricFeedback` to score
answers produced by an external LLM. Start a [vLLM](https://github.com/vllm-project/vllm)
server before launching training:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B \
  --host 127.0.0.1 --port 10001 --api-key keykey \
  --served-model-name feedback \
  --gpu-memory-utilization 0.5 --max-model-len 2k \
  --tensor-parallel-size 1
```

Run training or evaluation with the API-enabled feedback model:

```bash
python train_q_rag.py feedback.type=babilong_em \
  feedback.model=feedback feedback.use_api=true ...
```

All parameters in `configs/feedback/defaults.yaml`, such as sampling settings or
the served model name, can be modified directly in the file or overridden on the
CLI as shown above.

**🟥 Problems:**
The `configs/feedback/defaults.yaml` file defines a `vllm_config` dictionary that
is only used when launching vLLM in the same process (i.e. `feedback.use_api=False`), but this option doesn't work with training. 
To use vllm with training you need to use `feedback.use_api=True` and 
`vllm serve`. But in this case `vllm_config` is ignored; pass the same options as command
line flags to `vllm serve` instead.



