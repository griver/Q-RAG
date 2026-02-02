python ../train_q_rag.py envs=niah
python ../eval_retriever.py pretrained_path=<model_path> ++envs.test_env.dataset.split="test4k" num_samples=1000 seed=42 use_last=True
python eval_llm.py  <json_path>   --llm_name "Qwen/Qwen3-4B" --babi_task "niahmv" --max_tokens=512

