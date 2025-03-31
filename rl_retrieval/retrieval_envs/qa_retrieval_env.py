from typing import SupportsFloat, Any
import numpy as np
from gymnasium.core import ObsType
from typing import Callable
from rl_retrieval.retrieval_envs.retrieval_env import ARetrievalEnv
from torch.utils.data import Dataset

from dataloaders.localsets.musique import RetrievalMusique
from dataloaders.globalset import PATHS
from transformers import AutoTokenizer


class QARetrievalEnv(ARetrievalEnv):

    def __init__(self, reward_model, termination_func: Callable, max_steps: int):
        super().__init__(reward_model, termination_func, max_steps)
        self.chunks = []
        self.chunk_embeds = []
        self.sf_idx = []
        self.question = ""
        self.answer = ""
        self.chunks_mask = None
        self.retrieved_chunks = []
        self.last_action = None

    def _init_from_sample(self, sample):
        self.chunks = sample["chunks_texts"]
        self.chunk_embeds = sample.get("chunk_embeds")
        self.sf_idx = sample["sf_idx"]
        self.question = sample["question"]
        self.answer = sample["answer"]
        self.chunks_mask = np.ones(len(self.chunks), dtype=int)
        self.retrieved_chunks = []
        self.last_action = None

    def _make_obs(self):
        obs = {
            "question": self.question,
            "retrieved_chunks": [self.chunks[idx] for idx in self.retrieved_chunks],
            "chunks_mask": self.chunks_mask.copy(),
            "chunks": self.chunks,
            "chunk_embeds": self.chunk_embeds if self.chunk_embeds is not None else None,
        }
        return obs

    def step(self, action: list[int]) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = [idx for idx in action if self.chunks_mask[idx] == 1]

        self.retrieved_chunks = sorted(set(self.retrieved_chunks + action))
        self.chunks_mask[action] = 0
        self.last_action = action.copy()

        obs = self._make_obs()

        info = {
            "sf_idx": self.sf_idx.copy(),
            "max_steps": self.max_steps,
            "cur_step": self.cur_step,
            "answer": self.answer,
            "last_action": self.last_action.copy()
        }

        reward = self.reward_model.reward(obs, info)

        self.cur_step += 1

        terminated = self.termination_func(obs, info)
        truncated = self.cur_step >= self.max_steps

        return obs, reward, terminated, truncated, info

    def reset(self, sample, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(sample, seed)
        self.last_action = None
        info.update({
            "sf_idx": self.sf_idx.copy(),
            "max_steps": self.max_steps,
            "cur_step": self.cur_step,
            "answer": self.answer,
            "last_action": self.last_action
        })
        return obs, info

    def close(self):
        pass


class SimpleEnvAdapter(Dataset):
    """
    Simple adapter that adapts datasets Babilong, HotPotQA and MUSIQUE for QAREtreievalEnv.
    This adapter doesn't tokenize or embeds text chunks.

    You can create different adapter that for example tokenize every text in a sample or
    build faiss index over text chunks.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = self.dataset.name()

    def __getitem__(self, index):
        sample = self.dataset[index]
        question = sample["question"]
        if question.endswith("?"):
            question = question[:-1]

        sf_idx = []
        chunks_texts = []
        if self.dataset_name == 'hotpotqa':
            sp_title_set = set()
            sample_id = sample['_id']
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])

            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                chunk = title + " " + " ".join(sentences)
                chunks_texts.append(chunk)

        elif self.dataset_name == 'musique':
            sample_id = sample['id']
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                chunk = para['title'] + '. ' + para['paragraph_text']
                chunks_texts.append(chunk)

            # label order
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])

        elif self.dataset_name == 'babilong':
            sample_id = index
            for i, sent in enumerate(sample['chunks']):
                chunks_texts.append(sent)

            for i in sample['references_idx']:
                sf_idx.append(i)

        return {
            'id': sample_id,
            'question': question,
            'answer': sample["answer"],
            'chunks_texts': chunks_texts,
            'sf_idx': sf_idx,
        }


