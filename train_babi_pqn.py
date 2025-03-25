import sys
import os

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
from rl.pqn import PQN, PQNArgs
import time
import numpy as np
from collections import deque
from rl.babilong_env import BabilongEnv
from rl.sacd import SAC, SACArgs
from rl.text_env import TextReplayBuffer
from transformers import AutoModel, AutoTokenizer
from rl.bert_predictor import BertPredictor
from rl.text_env import TextEnv, TextReplayBuffer
from tqdm import tqdm
import torch


torch.set_default_device('cuda:1')
torch.set_float32_matmul_precision('high')

max_steps = 6
num_sentences = 50
task = "qa2_two-supporting-facts"
noise_train_path = "/home/nazar/pg19-train-with-sentences"
noise_path_test = "/home/nazar/pg19-test-with-sentences"
facts_train_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_train.txt"
facts_test_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_test.txt"

writer = SummaryWriter(comment="DQN_" + task)

fact_dataset = QA2FixWrapper(TaskDataset(facts_train_path), add_sentence_idx=True)  
test_fact_dataset = QA2FixWrapper(TaskDataset(facts_test_path), add_sentence_idx=True)   

noise_dataset = datasets.load_from_disk(noise_path_test)
noise_sampler = RetrSentenceSampler(noise_dataset)

dataset = RetrNoiseInjectionDataset(
    task_dataset=fact_dataset,
    noise_sentence_sampler=noise_sampler,
    num_sentences=num_sentences
)

dataset_test = RetrNoiseInjectionDataset(
    task_dataset=test_fact_dataset,
    noise_sentence_sampler=noise_sampler,
    num_sentences=num_sentences
)

bert_name = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True, revision="main")
bert_model = AutoModel.from_pretrained(bert_name, revision="main")

# bert_name_2 = "bert-base-uncased"
# tokenizer_2 = AutoTokenizer.from_pretrained(bert_name_2, use_fast=True, revision="main")
# bert_model_2 = AutoModel.from_pretrained(bert_name_2, revision="main")


action_embed = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()
action_embed_target = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()

state_embed = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()
state_embed_target = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()


agent = PQN(
    state_embed, action_embed,
    state_embed_target, action_embed_target
)


env = BabilongEnv( 
    embedder=agent.action_embed_target, 
    embed_tokenizer=tokenizer, 
    dataset=dataset,
    max_steps=max_steps
)

env_test = BabilongEnv( 
    embedder=agent.action_embed_target, 
    embed_tokenizer=tokenizer, 
    dataset=dataset_test,
    max_steps=max_steps
)

@torch.no_grad()
def evaluate(env_test, agent):
    s = env_test.reset()
    done = False
    a_embeds = env_test.get_extra_embeds(agent.critic.action_embed)
    r_sum = 0
    while not done:
        action, _, _ = agent.select_action(s, a_embeds, random=False, evaluate=True)
        s, _, reward, done = env_test.step(action)
        r_sum += reward
    
    return r_sum


buffer = TextReplayBuffer(32, tokenizer = tokenizer)
    
learning_start = 100

step = 0
R = 0
entropy_list = []
for ep_number in tqdm(range(300_000)):

    agent.policy.update(agent.critic)
    agent.v_net_target.update(agent.critic, agent.tau)
    agent.action_embed_target.update(agent.critic, agent.tau)

    s = env.reset()
    done = False

    agent.critic.action_embed.train()
    a_embeds = env.get_extra_embeds(agent.critic.action_embed)

    agent.policy.train()
    agent.critic.train()

    qf_loss, alpha_loss = 0, 0
    r_sum = 0

    a_all = []

    while not done:
        step += 1
        
        action, _, q_values  = agent.select_action(s, a_embeds, random = step < learning_start)
        s_next, a_data, reward, done = env.step(action)
        buffer.add(s, a_data, s_next, a_data, reward, done, 0, q_values.max().cpu().item())
        
        s = s_next
        R += reward
        r_sum += reward
        a_all.append(action)
        
        if step > learning_start and step % 32 == 0:
            for _ in range(1):
                s_batch, a_batch, next_s_batch, _, r_batch, not_done_batch, entropy_batch, q_batch = buffer.ordered_sample(30)
                qf_loss = agent.update(s_batch, a_batch, next_s_batch, q_batch, r_batch, not_done_batch)
                writer.add_scalar("qf_loss", qf_loss, step)
    
    writer.add_scalar("r_sum", r_sum, step)
    
    if ep_number % 100 == 0 and ep_number > 0:
        print("R", R / 100, "qf loss", qf_loss)
        print("alpha", agent.alpha)
        print(a_all, env.ref_ids)
        print(env.question, env.sentences[env.ref_ids[0]], env.sentences[env.ref_ids[1]])
        R = 0

    if ep_number % 100 == 0 and ep_number > 0:
        agent.policy.eval()
        agent.critic.action_embed.eval()

        r_eval = []
        
        for _ in range(100):
            r_eval.append(evaluate(env_test, agent))

        print("eval r_sum:", np.mean(r_eval))
        writer.add_scalar("eval r_sum", np.mean(r_eval), step)

