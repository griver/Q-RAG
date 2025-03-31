import numpy as np
from abc import ABC, abstractmethod


class RetrievalPolicy(ABC):

    @abstractmethod
    def act(self, obs, **kwargs):
        raise NotImplementedError()


class RNDPolicy(RetrievalPolicy):
    """
    Random policy chooses chunks randomly from list of available chunks.
    """
    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, obs, info=None):
        action_mask = obs['chunks_mask']
        available_ids = action_mask.nonzero()[0]
        chosen_actions = np.random.choice(available_ids, size=self.retrieve_k, replace=False)
        return chosen_actions

class OraclePolicy(RetrievalPolicy):
    """
    OraclePolicy uses info['sf_idx'] to select support facts.
    Actual agents should refrain from using info variable returned by the environment.
    """
    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, obs, info=None):
        sf_idx = info['sf_idx']
        choice = []
        for idx in sf_idx:
            if obs['chunks_mask'][idx] == 1.:
                choice.append(idx)

            if len(choice) == self.retrieve_k:
                break
        return choice
