from transformers import AutoTokenizer
from rl_retrieval.utils import all_facts_found
from rl_retrieval.feedback import GroundTruthFeedback
from rl_retrieval.retrieval_envs.qa_retrieval_env import QARetrievalEnv, SimpleEnvAdapter
from rl_retrieval.policy import RNDPolicy, OraclePolicy
from rl_retrieval.utils import all_facts_found
from dataloaders.localsets.musique import RetrievalMusique
from dataloaders.localsets.hotpotqa import RetrievalHotPotQA
from dataloaders.globalset import PATHS
import numpy as np


if __name__ == '__main__':
    seed = 42
    split = 'eval'
    num_samples = 1
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    #tokenizer here are used only to estimate length of samples in datasets

    dataset = RetrievalHotPotQA(
        path=PATHS['hotpotqa'], tokenizer=tokenizer, length=-1,
        min_context_len=0, max_context_len=1e7,
        type='any', anno_type='any', split=split, seed=seed
    )
    # dataset = RetrievalMusique(
    #     path=PATHS['hotpotqa'], tokenizer=tokenizer, length=-1,
    #     min_context_len=0, max_context_len=1e7,
    #     type='any', anno_type='any', split=split, seed=seed
    # )

    dataset = SimpleEnvAdapter(dataset)
    feedback_model = GroundTruthFeedback(per_fact_reward=0.05, completion_reward=1.)

    env = QARetrievalEnv(feedback_model, max_steps=4)
    terminated = truncated = False
    states = []
    rewards = []
    agent = OraclePolicy()

    for i in range(num_samples):
        step = 0
        print(f"\n################## START EPISODE #{i} ####################")
        print(f'=========== Step #{step} ===========')
        obs, info = env.reset(dataset[i])
        print(f"Question: {obs['question']}?")
        print('Relevant chunks:')
        for j in info['sf_idx']:
            print(f"Chunk #{j}:\n {obs['chunks'][j]}\n")


        while not (terminated or truncated):
            # generally agent should avoid using info but these are just examples:
            chunk_idx = agent.act(obs, info)
            print(f'Agent selected chunks: {chunk_idx}')
            obs, reward, terminated, truncated, info = env.step(chunk_idx)
            step += 1
            print(f'Reward: {reward:.2f}, terminated: {terminated}, truncated: {truncated}')
            print(f"chunk mask: {obs['chunks_mask']}")
            print('Retrieved chunks:\n', "\n".join(obs['retrieved_chunks']), "\n")
            print(f'=========== Step #{step} ===========')
            # log episode steps(obs, reward, terminated, truncated, info)
            # do some training
        print("################## END EPISODE #{i} ####################")