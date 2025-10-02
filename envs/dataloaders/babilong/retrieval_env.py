import numpy as np
import torch
import sys
import datasets
from transformers import AutoTokenizer
from envs.dataloaders.babilong.babilong_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rl.langchain_utils import ContrieverEmbeddingsAdapter
from envs.dataloaders.babilong.retrieval_babilong import RetrSentenceSampler, RetrievalBabiLong


def shuffle(noise, facts):
    N_facts = len(facts)
    N = len(noise) + N_facts
    facts_ids = sorted(np.random.choice(N, size=N_facts, replace=False))
    all = []
    noise_i, fact_i = 0, 0
    for i in range(N):
        if fact_i < N_facts and i == facts_ids[fact_i]:
            all.append(facts[fact_i])
            fact_i += 1
        else:
            all.append(noise[noise_i])
            noise_i += 1
    return all, facts_ids


class RetrievalEnv:
    def __init__(self,
                 embedder,
                 embed_tokenizer,
                 dataset=None,
                 reward_model=None,
                 max_steps=3,
                 done_when_rewarded=True,
                 ):
        super().__init__()
        self.done_when_rewarded=done_when_rewarded
        self.max_steps = max_steps
        self.max_sentence_len = 50
        self.max_batch_size = 64
        self.dataset = dataset

        self.embedder = embedder
        self.embed_tokenizer = embed_tokenizer
        self.rmodel = reward_model

        self.references = None
        self.question = None
        self.sentences = None
        self.facts_idx = None
        self.sent_embeds = None
        self.state = None
        self.available_acts = None
        self.chosen_sent_ids = None
        self.num_steps = 0

    def _init_from_sample(self, sample):
        self.references = list(sample['references'])
        self.question = sample['question']  # append as this is a single str
        self.answer = sample['answer']
        self.references_idx = sample.get('references_idx', None)
        self.sentences = np.asarray(sample['chunks'])
        self.facts_idx = list(sample['facts_idx'])
        self.sent_embeds = self.get_embeds(self.sentences)

    def reset(self, new_sample=None):
        if new_sample is not None:
            self._init_from_sample(new_sample)

        elif self.dataset is not None:
            N = len(self.dataset)
            i = np.random.randint(N)
            new_sample = self.dataset[i]
            self._init_from_sample(new_sample)

        if self.rmodel:
            self.rmodel.reset()

        self.num_steps = 0
        self.state = [self.question]
        self.available_acts = np.ones(len(self.sentences), dtype=bool)
        self.chosen_sent_ids = []
        return self._make_state()

    def step(self, chosen_acts):
        all_acts = set(self.chosen_sent_ids +list(chosen_acts))
        self.chosen_sent_ids = sorted(all_acts)
        retrieved_sentences = self.sentences[self.chosen_sent_ids]
        self.state = [self.question] + list(retrieved_sentences)
        self.available_acts[self.chosen_sent_ids] = False
        self.num_steps += 1

        done = self.num_steps >= self.max_steps
        r = self._reward()

        if self.done_when_rewarded and (r != 0.):
            done = True

        return self._make_state(), r, done

    @property
    def device(self):
        return self.embedder.device

    @torch.no_grad()
    def get_embeds(self, sentences):
        batch = self.embed_tokenizer(list(sentences), padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        B = batch["input_ids"].shape[0]
        embeds = []
        for i in range(0, B, self.max_batch_size):
            subbatch = {k:v[i:i+self.max_batch_size] for k, v in batch.items()}
            embeds.append(self.embedder(**subbatch).to("cpu"))

        embeds = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=0)
        return embeds


    def _make_state(self):
        s = [" ".join(s[:self.max_sentence_len] for s in self.state)]
        if len(s[0]) > 1024:
            print(f"State length is too big: L={len(s[0])}")

        state_embed = self.get_embeds(s)[0]

        return {
            'state': list(self.state),
            "acts": self.sentences,
            "acts_mask": self.available_acts.copy(),
            "state_embed": state_embed,
            "acts_embed": self.sent_embeds,
        }

    def _reward(self):
        if not self.rmodel:
            return 0.
        return self.rmodel.reward(self)

    def close(self):
        del self.sent_embeds

    def print_info(self):

        red = '\033[31m'
        end = '\033[m'
        print('Q:', red+self.question+end)
        print('References:')
        for r in self.references:
            print(r)
        print("A:", red+self.answer+end)
        print('Facts:')
        facts = self.sentences[self.facts_idx]
        for f in facts:
            if f in self.references:
                print(red+f+end)
            else:
                print(f)


class FaissRetrievalEnv(RetrievalEnv):

    def __init__(
            self,
            sample,
            embedder,
            embed_tokenizer,
            reward_model=None,
            max_steps=3,
    ):
        super(RetrievalEnv, self).__init__()
        self.max_steps = max_steps
        self.max_sentence_len = 50
        self.max_batch_size = 64

        self.embedder = embedder
        self.embed_tokenizer = embed_tokenizer
        self.embedder_adapter = ContrieverEmbeddingsAdapter(
            embedder, embed_tokenizer, self.max_batch_size
        )

        self.rmodel = reward_model
        self.sentences = []

        # print("Supporting facts:\n", sample['facts'])
        #self.sentences, _ = shuffle(background_text, sample['facts'])
        self.vector_store, self.facts_ids = self._make_vector_store(sample['noise'], sample['facts'])
        self.sentences.extend(sample['noise'])
        self.sentences.extend(sample['facts'])

        self.references = list(sample['references'])
        self.question = sample['question']  # append as this is a single str
        self.sentences = np.array(self.sentences)

        self.state = None
        self.available_acts = None
        self.chosen_sent_ids = None
        self.num_steps = 0

    def _make_vector_store(self, noise, facts):
        noise_embed = self.get_embeds(noise)
        vectorstore = FAISS.from_embeddings(
            zip(noise, noise_embed), self.embedder_adapter,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        facts_ids = self._add_to_vectorstore(vectorstore, facts)

        return vectorstore, facts_ids

    def _add_to_vectorstore(self, vectorstore, sentences):
        sent_embeds = self.get_embeds(sentences)
        facts_ids = vectorstore.add_embeddings(zip(sentences, sent_embeds))
        return facts_ids

    def _make_state(self):
        s = [" ".join(s[:self.max_sentence_len] for s in self.state)]
        if len(s[0]) > 1024:
            print(f"State length is too big: L={len(s[0])}")

        return {
            "state_str": s,
            "state": list(self.state),
            "vector_store": self.vector_store,
            "acts_text": self.sentences,
            "acts_mask": self.available_acts.copy(),
        }


class RetrievalPolicy:
    def act(self, state):
        raise NotImplementedError()


class RNDPolicy(RetrievalPolicy):
    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, state):
        action_mask = state['acts_mask']
        available_ids = action_mask.nonzero()[0]
        chosen_actions = np.random.choice(available_ids, size=self.retrieve_k, replace=False)
        return chosen_actions


class TopKExhaustiveSearch(RetrievalPolicy):
    """
    works only with original RetirevalEnv
    """
    def __init__(self, retrieve_k=1, epsilon=0.0):
        super().__init__()
        self.retrieve_k = retrieve_k
        self.epsilon=epsilon

    def act(self, state):
        a_mask = state['acts_mask']
        acts_ids = sorted(state['acts_mask'].nonzero()[0])

        if np.random.random() > self.epsilon:
            s_embed = state['state_embed']
            a_embed = state['acts_embed'][a_mask]
            scores = torch.inner(s_embed, a_embed)
            score_ids = torch.argsort(scores, descending=True)
            chosen_ids = [acts_ids[i] for i in score_ids[:self.retrieve_k]]
        else:
            chosen_ids = np.random.choice(acts_ids, size=self.retrieve_k, replace=False)
        return chosen_ids


class TopKFaiss(RetrievalPolicy):

    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, state):
        s_embed = state['state_embed']
        a_mask = state['acts_mask']
        a_embed = state['acts_embed'][a_mask]
        acts_ids = sorted(state['acts_mask'].nonzero()[0])
        scores = torch.inner(s_embed, a_embed)
        score_ids = torch.argsort(scores, descending=True)
        chosen_actions = [acts_ids[i] for i in score_ids[:self.retrieve_k]]
        return chosen_actions


class GroundTruthReward:
    def __init__(self, only_at_max_step=False):
        super().__init__()
        self.only_at_max_step = only_at_max_step

    def reward(self, env : RetrievalEnv, **kwargs):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        is_retrieved = []
        for r in env.references:
            is_retrieved.append(r in env.state)

        all_retrieved = all(is_retrieved)
        return float(all_retrieved)

    def reset(self):
        pass

class RewardForFacts:
    def __init__(self, reward_coef=1.):
        super().__init__()
        self.reward_coef = reward_coef

    def reward(self, env : RetrievalEnv, **kwargs):
        chosen_facts = set(env.chosen_sent_ids).intersection(env.facts_idx)
        n = len(chosen_facts) - self.prev_num_facts
        assert n >= 0, (f"How chat can happen?"
                        f" Selected facts: {env.chosen_sent_ids}, fact_ids: {env.facts_idx}, n={n}, prev_n={self.prev_num_facts}")
        self.prev_num_facts = len(chosen_facts)
        return float(n)*self.reward_coef

    def reset(self):
        self.prev_num_facts = 0


def evaluate(dataset, policy, retriever, retr_tokenizer, lm_tokenizer, max_steps=3):
    #policy = TopKPolicy(2)
    rewards = []
    N = len(dataset)
    for i in range(N):
        sample = dataset[i]
        env = RetrievalEnv(
            sample, retriever, retr_tokenizer,
            lm_tokenizer, GroundTruthReward(), max_steps=max_steps
        )
        s = env.reset()
        done = False
        reward = None
        while True:
            if done: break
            actions = policy.act(s)
            s, reward, done = env.step(actions)

        rewards.append(reward)
        retrieval_acc = np.mean(rewards)
        print(f"\rit {i+1}/{N}, retrieval accuracy: {retrieval_acc:.3f}", end="")

        del env

    retrieval_acc = np.mean(rewards)
    print(f"FINAL retrieval accuracy: {retrieval_acc:.3f}")
    return retrieval_acc


def play(policy, sample):

    print(sample.keys())
    env = RetrievalEnv(
        sample,
        embedder,
        embed_tokenizer,
        reward_model=GroundTruthReward(),
        max_steps=3
    )
    s = env.reset()
    print(s.keys())
    print("Question:", env.question)
    print("Sentences:")
    for i, sent in enumerate(env.sentences):
        sent_visual = sent.replace('\n', ' ')
        print(f"#{i}: {sent_visual}")
    print("num actions:", len(s['acts_mask']))

    done = False
    reward = 0.
    print("\n################## START EPISODE ####################")
    while True:
        print(f"step#{env.num_steps}")
        print("action mask:\n", s['acts_mask'].astype(np.int64))
        print(f"state: {' '.join(env.state)}")
        print("reward:", reward)
        # print("state_embed:", s['state_embed'].shape)

        if done:
            print("DONE!")
            break

        actions = policy.act(s)
        print("selected actions:", actions)
        s, reward, done = env.step(actions)

    print("#########################################")


def print_dict(d, indent=0):
    indent_str = " "*indent
    for k,v in d.items():
        print(indent_str,f"KEY: {k}")

        if isinstance(v, dict):
            print(indent_str, "dict{")
            print_dict(d, indent+2)
            print(indent_str, "}")

        elif hasattr(v, "__len__") and not isinstance(v, str):
             print(indent_str, "list[")
             for i, e in enumerate(v):
                 print(indent_str, f"  {i}:", e.replace("\n", ""))
             print(indent_str, "]")
        else:
            print(indent_str, "VALUE:", v.replace("\n", ""))


if __name__ == "__main__":

    use_retrieval_dataset = True
    num_tokens = 16000  #measure size of sample in tokens if use_retrieval_dataset == False
    num_sentences = 10 #measure size of sample in sentences if use_retrieval_dataset == True
    task = "qa2_two-supporting-facts"

    train_path = f"data/tasks_1-20_v1-2/en-10k/{task}_train.txt"
    test_path = f"data/tasks_1-20_v1-2/en-10k/{task}_test.txt"


    # background text
    lm_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    contriever_path = "../contriever"
    if contriever_path not in sys.path:
        sys.path.append(contriever_path)
    from src.contriever import Contriever
    from transformers import AutoTokenizer

    device = torch.device('cuda:0')
    embedder = Contriever.from_pretrained("facebook/contriever").to(device)
    embed_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    #task_dataset_train = TaskDataset(train_path,) #max_n_facts=10)
    task_dataset_test = TaskDataset(test_path,) #max_n_facts=10)


    if use_retrieval_dataset:
        noise_dataset_name = "data/pg19-test-with-sentences"
        noise_dataset = datasets.load_from_disk(noise_dataset_name)
        noise_sampler_test = RetrSentenceSampler(noise_dataset)
        dataset_test = RetrievalBabiLong(task_dataset=task_dataset_test,
                                         noise_sentence_sampler=noise_sampler_test,
                                         num_sentences=num_sentences)
        print(f'retrieval dataset is loaded')
        sample = dataset_test[0]
        #print("first sample:")
        #print_dict(sample)
        #env = FaissRetrievalEnv(sample, embedder, embed_tokenizer, )
        policy = TopKExhaustiveSearch(1)
        play(policy, sample)

    else:
        noise_dataset_name = "pg19"
        noise_dataset = datasets.load_dataset(noise_dataset_name)
        noise_sampler_test = SentenceSampler(noise_dataset['test'], tokenizer=lm_tokenizer)

        dataset_test = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                             noise_sampler=noise_sampler_test,
                                             tokenizer=lm_tokenizer,
                                             sample_size=num_tokens)

        policy = TopKExhaustiveSearch(1)
        evaluate(dataset_test, policy, embedder, embed_tokenizer, lm_tokenizer, max_steps=10)
        #play(policy, dataset_train[0])
