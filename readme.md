# Multi-Step Retrieval for Long Context

## Overview
Folders: 
* `beam_retriever` contains beam retriever baseline
* `dataloades` contains code to run experiments on datasets 'Babilong', 'HotPotQA', 'Musique', etc.
* `prompts_and_metrics` contains prompts and metrics to evaluate llm on our datasets
* `rl_retieval` contains first tamplates formulating interaction between `RetievalEnv` and `RetrievalPolicy`

## Quickstart
Run `test_env.py` to see simple example of interaction between `RetievalEnv` and a toy policy:
```python
python rl_retrieval/test_env.py
```