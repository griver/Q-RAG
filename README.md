# Multi-Step Retrieval via Reinforcement Learning 


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
Example – HotpotQA:
```bash
python train_q_rag.py envs=hotpotqa max_action_length=140 envs.max_steps=3 batch_size=16 accumulate_grads=2 eval_episodes=100
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



