from typing import SupportsFloat, Any
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from abc import ABC, abstractmethod


class ARetrievalEnv(Env):

    def __init__(
            self,
            feedback_model,
            #termination_func,
            max_steps
    ):
        """
        :param dataset: QA dataset
        :param feedback_model: an instance that will return reward given current state
        :param max_steps: maximum number of retrieval steps in episode
        :param termination_func: determines if episode is done before reaching max_steps
        :param top_k: number of chunks to return per retrieval step
        """
        super().__init__()
        self.fb_model = feedback_model
        self.max_steps = max_steps
        #self.termination_func = termination_func
        #self.top_k = top_k
        self.state = None
        self.cur_step = 0


    @abstractmethod
    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass
        #updates current state and returns new_state, reward, done, truncated, extra_info
        # self.cur_step += 1
        # truncated = (self.cur_step >= self.max_steps)
        # obs = self._make_obs()
        # reward = self.reward_model.reward(self)
        # done = self.done_func(self)
        # return obs, reward, done, truncated, None

    def reset(self, sample, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        self.state = None
        self.cur_step = 0
        self._init_from_sample(sample)
        obs = self._make_obs()
        info = {}
        self.fb_model.reset(obs, info)
        return obs, info

    @abstractmethod
    def _init_from_sample(self, sample):
        pass

    @abstractmethod
    def _make_obs(self):
        pass

    @abstractmethod
    def close(self):
        pass
