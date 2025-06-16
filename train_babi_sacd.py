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
import numpy as np
from envs.babilong_env import BabilongEnv
from rl.agents.sacd import SAC, SACArgs
# from rl.sarsa import SARSA, SARSAArgs
from transformers import AutoModel, AutoTokenizer
from rl.bert_predictor import BertPredictor
from rl.text_env import TextReplayBuffer
from tqdm import tqdm

max_steps = 10
num_sentences = 50
task = "qa2_two-supporting-facts"
noise_train_path = "/home/nazar/pg19-train-with-sentences"
noise_path_test = "/home/nazar/pg19-test-with-sentences"
facts_train_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_train.txt"
facts_test_path = f"/home/nazar/tasks_1-20_v1-2/en-10k/{task}_test.txt"

writer = SummaryWriter(comment="SACD_" + task)

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


action_embed = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()
action_embed_target = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()

state_embed = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()
state_embed_target = BertPredictor(bert_model, 12, tokenizer, 768, 256, 1).cuda()

# agent = SARSA(
#     state_embed, action_embed, state_embed_target, action_embed_target, 
#     SARSAArgs(gamma=0.99, tau=0.01,  lr=5e-5, max_steps=100_000 // 4)
# )

agent = SAC(
    state_embed, action_embed, state_embed_target, action_embed_target, 1,
    SACArgs(gamma=0.99, tau=0.01, lr=5e-5, alpha=0.005, automatic_entropy_tuning=True, max_steps=(100_000 // 4) * max_steps)
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

buffer = TextReplayBuffer(100_000, tokenizer = tokenizer)


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
        
        action, logp, e = agent.select_action(s, a_embeds, random = step < learning_start)
        s_next, a_data, reward, done = env.step(action)
        buffer.add(s, a_data, s_next, reward, done, e)
        entropy_list.append(e)
        if len(entropy_list) > 100:
           entropy_list = entropy_list[-100:]
 
        s = s_next
        R += reward
        r_sum += reward
        a_all.append(action)
        
        if step > learning_start and step % 4 == 0:
            s_batch, a_batch, next_s_batch, r_batch, not_done_batch, entropy_batch = buffer.sample(16)
            qf_loss, alpha_loss = agent.update(s_batch, a_batch, next_s_batch, r_batch, not_done_batch, np.mean(entropy_list), step)
            writer.add_scalar("qf_loss", qf_loss, step)
            writer.add_scalar("alpha", agent.alpha, step)
    
    writer.add_scalar("r_sum", r_sum, step)
    
    if ep_number % 100 == 0 and ep_number > 0:
        print(R / 100, qf_loss, agent.alpha, alpha_loss)
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

