from abc import ABC, abstractmethod
from  typing import  Dict, Union, List

class AFeedbackModel(ABC):
    FEEDBACK_MODEL_NAME: str

    @abstractmethod
    def __init__(self, never_terminate=False):
        self.completed = False
        self.never_terminate = never_terminate

    @abstractmethod
    def reset(self, obs: dict | List[dict], info: dict | List[dict]) -> None:
        """
        :param obs: dict in case of single env, or a list of dicts in batch case
        :param info: dict in case of single env, or a list of dicts in batch case
        """
        if isinstance(obs, dict):
            self.completed = False
        else:
            self.completed = [False for _ in obs]

    def get_feedback(
            self,
            obs: dict | List[dict],
            info: dict | List[dict],
            truncated: bool | List[bool]=False
    ) -> dict | List[dict]:
        """
        :param obs: dict in case of single env, or a list of dicts in batch case
        :param info: dict in case of single env, or a list of dicts in batch case
        :param truncated: True if episode is exceeded maximum number of steps, list of bools in batch case
        :return: a dict containing information about rewards and termination of the episode.
        """
        reward = self.reward(obs, info, is_final=truncated)

        if isinstance(self.completed, bool):
            terminated = False if self.never_terminate else self.completed
            return {
                'reward': reward,
                'terminated': terminated,
            }
        else:
            assert len(obs) == len(info) == len(self.completed), f'expected a batch of {len(self.completed)} observations'
            terminated = [False]*len(self.completed) if self.never_terminate else list(self.completed)
            return [{'reward': r, 'terminated': done} for r, done in zip(reward, terminated)]

    @abstractmethod
    def reward(
            self,
            obs: dict | List[dict],
            info: dict | List[dict],
            is_final: bool| List[bool]=False
    ) -> float | List[float]:
        pass

    @abstractmethod
    def copy(self):
        pass


class DummyFeedbackModel(AFeedbackModel):
    """
    Dummy feedback model. Always return 0. reward, never terminates an episode before truncation.
    Maybe usefull for parallel env execution.
    """
    FEEDBACK_MODEL_NAME: "dummy"
    def __init__(self):
        super().__init__(never_terminate=True)

    def reward(self, obs, info, is_final=None):
        return 0.

    def copy(self):
        return DummyFeedbackModel()


class GroundTruthFeedback(AFeedbackModel):
    """
    This version takes into account position of the support facts.
    In babi tasks several events could have completely identical text descriptions,
    but only one of them can be considered a support fact/reference fact.

    I.E. Merry could visit the same location several times.
    But only the last event allows us to tell where she is at the end of the story.

    This reward takes into account temporal information that allows to distinguish
    true support facts, from similar events.
    """
    FEEDBACK_MODEL_NAME="ground_truth"
    def __init__(self, penalize_extra_steps=False, completion_reward=1.0, per_fact_reward=0.0, never_terminate=False):
        super().__init__(never_terminate=never_terminate)
        #r_0 = 0.1, r_1 = 1.1
        self.per_fact_reward = per_fact_reward
        self.completion_reward = completion_reward
        self.penalize_extra_steps = penalize_extra_steps
        self.sf_idx = None
        self.found_facts = set()

    def reset(self, obs, info) -> None:
        super().reset(obs, info)
        self.sf_idx = None
        self.found_facts.clear()
        self.completed = False

    def reward(self, obs, info, is_final=False) -> float:
        #this reward doesn't care if the step was final or not
        if self.sf_idx is None:
            self.sf_idx = set(info["sf_idx"])

        pred_idx = set(obs['pred_idx'])
        new_facts = pred_idx.intersection(self.sf_idx) - self.found_facts
        step_r = len(new_facts) * self.per_fact_reward

        self.found_facts.update(new_facts)

        term_r = 0.
        if not self.completed and self.sf_idx.issubset(self.found_facts):
            self.completed = True
            if self.penalize_extra_steps:
                term_r = (0.5 + 0.5 * len(self.sf_idx) / (len(pred_idx) + 1e-5)) * self.completion_reward
            else:
                term_r = self.completion_reward

        return term_r + step_r

    def copy(self):
        return GroundTruthFeedback(
            penalize_extra_steps=self.penalize_extra_steps,
            completion_reward=self.completion_reward,
            per_fact_reward=self.per_fact_reward,
        )

