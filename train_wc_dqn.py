import sys
import os

from rl.sarsa import SARSA, SARSAArgs
from rl.words_counter_env import WordsCounterEnv


repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)


from babilong_fix import QA2FixWrapper
from rl.retrieval_babilong import RetrNoiseInjectionDataset, RetrSentenceSampler
from babilong_utils import TaskDataset
from torch.utils.tensorboard import SummaryWriter
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import torch
import sys
import time
import numpy as np
from collections import deque
from rl.babilong_env import BabilongEnv
from rl.sacd import SAC, SACArgs
# from rl.sarsa import SARSA, SARSAArgs
from rl.text_env import TextReplayBuffer
from transformers import AutoModel, AutoTokenizer
from rl.bert_predictor import BertPredictor
from rl.text_env import TextEnv, TextReplayBuffer
from tqdm import tqdm
import torch


# torch.set_default_device('cuda:1')
max_steps = 10
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
print("loading dataset: AIRI-NLP/quality_counter_new_1024")
dataset = load_dataset("AIRI-NLP/quality_counter_new_1024")

writer = SummaryWriter(comment="DQN_" + "AIRI-NLP/quality_counter_new_1024")

bert_name = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True, revision="main")
bert_model = AutoModel.from_pretrained(bert_name, revision="main")


action_embed = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()
action_embed_target = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()

state_embed = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()
state_embed_target = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()

agent = SARSA(
    state_embed, action_embed, state_embed_target, action_embed_target, 
    SARSAArgs(gamma=0.99, tau=0.01,  lr=5e-5, max_steps=(100_000 // 4) * max_steps)
)

env = WordsCounterEnv(
    dataset, 
    block_size=16, 
    embedder=agent.action_embed_target, 
    max_length=1024, 
    embed_tokenizer=tokenizer, 
    max_steps_count=max_steps,
    add_question=True,
    subset="train"
)

env_test = WordsCounterEnv(
    dataset, 
    block_size=16, 
    embedder=agent.action_embed_target, 
    max_length=1024, 
    embed_tokenizer=tokenizer, 
    max_steps_count=max_steps,
    add_question=True,
    subset="test"
)

buffer = TextReplayBuffer(100_000, tokenizer = tokenizer)


def evaluate(env_test, agent):
    s = env_test.reset()
    done = False
    a_embeds = env_test.get_extra_embeds(agent.critic.action_embed)
    r_sum = 0
    while not done:
        action = agent.select_action(s, a_embeds, random=False, evaluate=True)
        s, _, reward, done = env_test.step(action)
        r_sum += reward
    
    return int(r_sum >= len(env_test.word_positions))
    

learning_start = 2_000

step = 0
R = 0
entropy_list = []
for ep_number in tqdm(range(100_000)):

    s = env.reset()
    done = False
    a_embeds = env.get_extra_embeds(agent.critic.action_embed)
    agent.policy.update(agent.critic)
    agent.policy.train()
    agent.critic.action_embed.train()

    qf_loss, alpha_loss = 0, 0
    r_sum = 0

    a_all = []

    while not done:
        step += 1
        
        action = agent.select_action(s, a_embeds, random = step < learning_start)
        s_next, a_data, reward, done = env.step(action)
        buffer.add(s, a_data, s_next, reward, done, 0)
        
        s = s_next
        R += reward
        r_sum += reward
        a_all.append(action)
        
        if step > learning_start and step % 4 == 0:
            s_batch, a_batch, next_s_batch, r_batch, not_done_batch, entropy_batch = buffer.sample(32)
            qf_loss = agent.update(s_batch, a_batch, next_s_batch, r_batch, not_done_batch, step)
            writer.add_scalar("qf_loss", qf_loss, step)
    
    writer.add_scalar("r_sum", r_sum, step)
    
    if ep_number % 100 == 0 and ep_number > 0:
        print(R / 100, qf_loss)
        print(a_all, env.word_positions)
        R = 0

    if ep_number % 100 == 0 and ep_number > 0:
        agent.policy.eval()
        agent.critic.action_embed.eval()

        r_eval = []
        
        for _ in range(100):
            r_eval.append(evaluate(env_test, agent))

        print("eval r_sum:", np.mean(r_eval))
        writer.add_scalar("eval r_sum", np.mean(r_eval), step)


        r_eval = []
        
        for _ in range(100):
            r_eval.append(evaluate(env, agent))

        print("train r_sum:", np.mean(r_eval))
        writer.add_scalar("train r_sum", np.mean(r_eval), step)


