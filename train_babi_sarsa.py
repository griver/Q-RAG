import sys
import os

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)


from envs.babilong.babilong_fix import QA2FixWrapper
from envs.babilong.retrieval_babilong import RetrievalBabiLong, RetrSentenceSampler
from envs.babilong.babilong_utils import TaskDataset
from torch.utils.tensorboard import SummaryWriter
import datasets
from rl.sarsa import SARSA, SARSAArgs
import numpy as np
from rl.babilong_env import BabilongEnv
from transformers import AutoModel, AutoTokenizer
from rl.bert_predictor import BertPredictor
from rl.text_env import TextReplayBuffer
from tqdm import tqdm
import torch


torch.set_default_device('cuda:0')
torch.set_float32_matmul_precision('high')

max_steps = 6
num_sentences = 50
task = "qa2_two-supporting-facts"
noise_train_path = "/home/nazar/pg19-train-with-sentences"
noise_path_test = "/home/nazar/pg19-test-with-sentences"
facts_train_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_train.txt"
facts_test_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_test.txt"

writer = SummaryWriter(comment="SARSA_" + task)

fact_dataset = QA2FixWrapper(TaskDataset(facts_train_path), add_sentence_idx=True)  
test_fact_dataset = QA2FixWrapper(TaskDataset(facts_test_path), add_sentence_idx=True)   

noise_dataset = datasets.load_from_disk(noise_path_test)
noise_sampler = RetrSentenceSampler(noise_dataset)

dataset = RetrievalBabiLong(
    task_dataset=fact_dataset,
    noise_sentence_sampler=noise_sampler,
    num_sentences=num_sentences
)

dataset_test = RetrievalBabiLong(
    task_dataset=test_fact_dataset,
    noise_sentence_sampler=noise_sampler,
    num_sentences=num_sentences
)

bert_name = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True, revision="main")
bert_model = AutoModel.from_pretrained(bert_name, revision="main")


action_embed = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()
action_embed_target = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()

state_embed = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()
state_embed_target = BertPredictor(bert_model, 6, tokenizer, 768, 256, 1).cuda()

agent = SARSA(
    state_embed, action_embed, state_embed_target, action_embed_target, 
    SARSAArgs(lr=1e-4, max_steps=(40_000 // 4) * max_steps, warmup_steps=1000, exploration_steps=10000,
            epsilon_warmup=0.5, epsilon_final=0.01)
)


env = BabilongEnv( 
    embedder=agent.critic_target.action_embed, 
    embed_tokenizer=tokenizer, 
    dataset=dataset,
    max_steps=max_steps
)

env_test = BabilongEnv( 
    embedder=agent.critic_target.action_embed, 
    embed_tokenizer=tokenizer, 
    dataset=dataset_test,
    max_steps=max_steps
)

buffer = TextReplayBuffer(100_000, tokenizer = tokenizer)

def evaluate(env_test, agent):
    s = env_test.reset()
    done = False
    a_embeds = env_test.get_extra_embeds(agent.critic.action_embed)
    r_sum = 0
    while not done:
        action = agent.select_action(s, a_embeds, evaluate=True)
        s, _, reward, done = env_test.step(action)
        r_sum += reward
    
    return r_sum
    

learning_start = 100

step = 0
R = 0
entropy_list = []
for ep_number in tqdm(range(40_000)):

    s = env.reset()
    done = False
    a_embeds = env.get_extra_embeds(agent.critic.action_embed)
    agent.policy.update(agent.critic)
    agent.policy.eval()
    agent.critic.train()

    qf_loss, alpha_loss = 0, 0
    r_sum = 0

    s_all, a_all, a_data_all, next_s_all, r_all, dones_all, entropy_all = [], [], [], [], [], [], []

    while not done:
        step += 1
        
        action = agent.select_action(s, a_embeds, evaluate=False)
        s_next, a_data, reward, done = env.step(action)
        # buffer.add(s, a_data, s_next, reward, done, 0)

        s_all.append(s)
        a_data_all.append(a_data)
        a_all.append(action)
        next_s_all.append(s_next)
        r_all.append(reward)
        dones_all.append(done)
        entropy_all.append(0)
        
        s = s_next
        R += reward
        r_sum += reward
        
        if step > learning_start and step % 4 == 0:
            s_batch, a_batch, next_s_batch, next_a_batch, r_batch, not_done_batch, entropy_batch = buffer.sample(32)
            qf_loss = agent.update(s_batch, a_batch, next_s_batch, next_a_batch, r_batch, not_done_batch)
            writer.add_scalar("qf_loss", qf_loss, step)

    buffer.add_episode(s_all, a_data_all, next_s_all, r_all, dones_all)
    
    writer.add_scalar("r_sum", r_sum, step)
    
    if ep_number % 100 == 0 and ep_number > 0:
        print("R", R / 100, "qf loss", qf_loss)
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

