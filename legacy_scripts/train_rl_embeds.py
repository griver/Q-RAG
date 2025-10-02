from rl.replay_buffer import TextReplayBuffer
from envs.dataloaders.babilong.retrieval_babilong import RetrievalBabiLong, RetrSentenceSampler
from rl.agents.agent import RetrievalAgent
from envs.dataloaders.babilong.retrieval_env import RetrievalEnv, RetrievalPolicy, TopKExhaustiveSearch, RewardForFacts
from envs.dataloaders.babilong.babilong_utils import TaskDataset
import datasets
import torch
import sys
import time
import numpy as np
from collections import deque


def load_sentence_embedder():
    contriever_path = "../contriever"
    if contriever_path not in sys.path:
       sys.path.append(contriever_path)
    from src.contriever import Contriever
    from transformers import AutoTokenizer
    encoder = Contriever.from_pretrained("facebook/contriever").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    return encoder, tokenizer


def create_babilong_env(fact_path, noise_path, num_sentences=10, max_steps=3, done_when_rewarded=False):
    fact_dataset = TaskDataset(fact_path)  # max_n_facts=10)

    noise_dataset = datasets.load_from_disk(noise_path)
    noise_sampler = RetrSentenceSampler(noise_dataset)

    dataset = RetrievalBabiLong(
        task_dataset=fact_dataset,
        noise_sentence_sampler=noise_sampler,
        num_sentences=num_sentences
    )
    act_encoder, act_tokenizer = load_sentence_embedder()
    env = RetrievalEnv(
        act_encoder,
        act_tokenizer,
        dataset,
        max_steps=max_steps,
        reward_model=RewardForFacts(),
        done_when_rewarded=done_when_rewarded
    )
    return env


def discounted_returns(rewards, next_value=0, gamma=0.99):
    """
    Computes discounted n-step returns for rollout. Expects tensors or numpy.arrays as input parameters
    The function doesn't detach tensors, so you have to take care of the gradient flow by yourself.
    :return:
    """
    rollout_steps = len(rewards)
    returns = np.zeros(rollout_steps, dtype=np.float32)  # [None] * rollout_steps
    R = next_value
    for t in reversed(range(rollout_steps)):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns


def train(
        policy :RetrievalPolicy,
        env: RetrievalEnv,
        buffer: TextReplayBuffer,
        agent: RetrievalAgent,
        batch_size=32,
        train_every = 4,
        eval_every = 1000,
        warmup=500,
        gamma=0.99,
        logdir=None,
    ):

    final_r = []
    ep_len = []
    losses = deque(maxlen=50)
    num_episodes = 0
    step = 0
    best_eval = 0.
    while step < buffer.max_size:

        info = env.reset()
        s = [info['state'],]
        a = []
        r = []
        dones = []
        for t in range(env.max_steps):
            step += 1
            a_id = policy.act(info)
            if len(a_id) > 1: raise ValueError("only single retrieval per step, for now")
            new_info, reward, done = env.step(a_id)

            s.append(new_info['state'])
            a.append(info['acts'][a_id[0]])
            r.append(reward)
            dones.append(done)

            if done:
                num_episodes += 1

                ep_len.append(t+1)
                final_r.append(sum(r))
                rtg = discounted_returns(r, gamma=gamma)
                buffer.add_episode(s[:-1], a, s[1:], rtg, dones)
                #if reward == 0:
                #    print_episode(env, s, a, rtg, dones)

            info = new_info

            if step > max(warmup, batch_size*5) and (step % train_every == 0):
                batch = buffer.sample(batch_size)
                loss = agent.update(*batch)
                losses.append(loss)
                print(
                    f"\r#ep={num_episodes}: E[T]={np.mean(ep_len):.2f}, E[R]={np.mean(final_r):.2f}, loss={np.mean(losses):.3f}",
                    end=''
                )

            if step % eval_every == 0 and (step >= warmup):
                eval_r, _ = evaluate(env, agent, max_episodes=10)
                if eval_r >= best_eval:
                    best_eval = eval_r
                    if logdir: agent.save(logdir, step)

            if done: break

def print_episode(env, states, acts, rtg, dones):
    T = len(acts)
    print(f"============== Episode len: {len(acts)}, ==================")
    env.print_info()
    print('------------------------------------')
    for i in range(T):
        print(f"S[{i}]: {states[i]}, r[{i}]={rtg[i]} act[{i}]={acts[i]}")
        print("===============================")
    input(">=")


def evaluate(env, agent, max_episodes=100):
    dataset = env.dataset
    N = min(len(env.dataset), max_episodes)
    final_r = []
    ep_len = []
    for i in range(N):
        info = env.reset(dataset[i])
        s = [info['state'], ]
        a = []
        r = []
        dones = []
        for t in range(env.max_steps):
            a_id = agent.act(info)
            if len(a_id) > 1: raise ValueError("only single retrieval per step, for now")
            new_info, reward, done = env.step(a_id)

            s.append(new_info['state'])
            a.append(info['acts'][a_id[0]])
            r.append(reward)
            dones.append(done)
            info = new_info
            if done:
                ep_len.append(t + 1)
                final_r.append(sum(r))
                break

    print(f"\n EVALUATION: E[T]={np.mean(ep_len):.2f}. E[R]={np.mean(final_r):.2f}")
    print(f"Q: {s[0]}")
    for i, (a_i, r_i) in enumerate(zip(a, r)):
        print(f"#{i}: a={a_i}, r={r_i}")
    return np.mean(final_r), ep_len

def process_facts(sample):
    from nltk.tokenize import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    facts = sample['facts']
    q = sample['question']
    words = tokenizer.tokenize(q)
    item = words[-2]
    print(f"QUESTION ITEM: {item}")
    print("FACTS:")
    for f in facts:
        if item in f:
            print(f)
    print("RELEVANT FACTS:")
    for f in sample['references']:
        print(f)
    print(f'ANSWER: {sample["answer"]}')
    print('=========================================\n')

if __name__ == "__main__":
    device = torch.device('cuda:0')
    max_steps = 6
    num_sentences = 50 #measure size of sample in sentences if use_retrieval_dataset == True
    task = "qa2_two-supporting-facts"
    noise_train_path = "data/pg19-train-with-sentences"
    noise_path_test = "../data/babilong/pg19-test-with-sentences"
    facts_train_path = f"data/tasks_1-20_v1-2/en-10k/{task}_train.txt"
    facts_test_path = f"data/tasks_1-20_v1-2/en-10k/{task}_test.txt"

    env = create_babilong_env(facts_test_path, noise_path_test, num_sentences, max_steps=max_steps)

    # for i in range(10):
    #     process_facts(env.dataset[i])
    #
    # exit()

    agent = RetrievalAgent(env.embedder, env.embed_tokenizer)
    replay_buffer = TextReplayBuffer(max_size=100000)#10000)
    policy = TopKExhaustiveSearch(1, epsilon=0.5)
    train(policy, env, replay_buffer, agent,
          batch_size=16, warmup=500, eval_every=500, gamma=0.1,
          logdir=f"runs/{task}-t{max_steps}-sent{num_sentences}/{time.strftime('%Y%m%d-%H%M')}/"
    )
    #task_dataset_train = TaskDataset(train_path,) #max_n_facts=10)
    #print("first sample:")
    #print_dict(sample)
    #env = FaissRetrievalEnv(sample, embedder, embed_tokenizer, )


        #play(policy, dataset_train[0])
