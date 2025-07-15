import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import FlaxAutoModel, AutoTokenizer, FlaxBertPreTrainedModel
from flax import linen as nn
from flax.struct import PyTreeNode
from flax import nnx
from flax.nnx import bridge

repo_dir = os.path.dirname(os.path.abspath("../"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)

from envs.dataloaders.babilong.babilong_fix import QA2FixWrapper
from envs.dataloaders.babilong.retrieval_babilong import RetrievalBabiLong, RetrSentenceSampler
from envs.dataloaders.babilong.babilong_utils import TaskDataset
from torch.utils.tensorboard import SummaryWriter
import datasets
from rl.flax_dqn import FlaxDQN, DQNArgs
import numpy as np
from envs.qa_env import QAEnv
from rl.jax_text_env import TextReplayBuffer
from tqdm import tqdm


# torch.set_default_device('cuda:1')

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

model_name = "facebook/contriever"
bert_model: FlaxBertPreTrainedModel = FlaxAutoModel.from_pretrained('facebook/contriever', revision="main", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', use_fast=True, revision="main", clean_up_tokenization_spaces=True)


agent = FlaxDQN(
    bert_model, 
    DQNArgs(gamma=0.99, tau=0.01,  lr=5e-5, max_steps=(20_000 // 4) * max_steps)
)


env = QAEnv(
    embedder=agent.action_embed_target, 
    embed_tokenizer=tokenizer, 
    dataset=dataset,
    max_steps=max_steps,
    max_embed_length=64
)

env_test = QAEnv(
    embedder=agent.action_embed_target, 
    embed_tokenizer=tokenizer, 
    dataset=dataset_test,
    max_steps=max_steps,
    max_embed_length=64
)

buffer = TextReplayBuffer(100_000, tokenizer = tokenizer)


def evaluate(env_test, agent):
    agent.action_embed.eval()
    agent.policy.eval()

    s = env_test.reset()
    done = False
    a_embeds = env_test.get_extra_embeds(agent.action_embed)
    r_sum = 0
    while not done:
        action = agent.select_action(s, a_embeds, random=False, evaluate=True)
        s, _, reward, done = env_test.step(action)
        r_sum += reward
    
    return r_sum
    

learning_start = 2_000

step = 0
R = 0
entropy_list = []
for ep_number in tqdm(range(20_000)):

    s = env.reset()
    done = False
    q_model = agent.get_q_model()
    a_embeds = env.get_extra_embeds(agent.action_embed)
    agent.policy.update(q_model)
    agent.policy.train()

    qf_loss, alpha_loss = 0, 0
    r_sum = 0

    a_all = []

    while not done:
        step += 1
        
        action = agent.select_action(s, a_embeds, random = step < learning_start or step % 10 == 0)
        s_next, a_data, reward, done = env.step(action)
        buffer.add(s, a_data, s_next, reward, done, 0)
        
        s = s_next
        R += reward
        r_sum += reward
        a_all.append(action)
        
        if step > learning_start and step % 4 == 0:
            s_batch, a_batch, next_s_batch, r_batch, not_done_batch, entropy_batch = buffer.sample(16)
            qf_loss = agent.update(s_batch, a_batch, next_s_batch, r_batch, not_done_batch)
            # writer.add_scalar("qf_loss", np.asarray(qf_loss), step)
    
    if ep_number % 100 == 0 and ep_number > 0:
        writer.add_scalar("r_sum", r_sum, step)
        print(R / 100, qf_loss)
        print(a_all, env.ref_ids)
        print(env.question, env.sentences[env.ref_ids[0]], env.sentences[env.ref_ids[1]])
        R = 0

    if ep_number % 100 == 0:
        r_eval = []
        
        for _ in range(100):
            r_eval.append(evaluate(env_test, agent))

        print("eval r_sum:", np.mean(r_eval))
        writer.add_scalar("eval r_sum", np.mean(r_eval), step)

