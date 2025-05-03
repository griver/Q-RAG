from abc import ABC, abstractmethod

class AFeedbackModel(ABC):

    @abstractmethod
    def __init__(self):
        self.completed = False

    @abstractmethod
    def reset(self, obs, info):
        self.completed = False

    def get_feedback(self, obs, info, truncated=False) -> dict:
        """
        :param obs:
        :param info:
        :param truncated: True if episode is exceeded maximum number of steps
        :return: a dict containing information about rewards and termination of the episode.
        """
        reward = self.reward(obs, info, is_final=truncated)
        terminated = self.completed
        return {
            'reward': reward,
            'terminated': terminated,
        }

    @abstractmethod
    def reward(self, obs, info, is_final=None):
        pass


class GroundTruthFeedback(AFeedbackModel):

    def __init__(self, per_fact_reward=0.1, completion_reward=1.0):
        super().__init__()
        #r_0 = 0.1, r_1 = 1.1
        self.per_fact_reward = per_fact_reward
        self.completion_reward = completion_reward
        self.sf_idx = None
        self.found_facts = set()

    def reset(self, obs, info) -> None:
        super().reset(obs, info)
        self.sf_idx = None
        self.found_facts.clear()
        self.completed = False

    def get_feedback(self, obs, info, truncated=False):
        reward = self.reward(obs, info, is_final=truncated)
        terminated = self.completed
        return {
            'reward': reward,
            'terminated': terminated,
        }

    def reward(self, obs, info, is_final=None) -> float:
        #this reward doesn't care if the step was final or not
        if self.sf_idx is None:
            self.sf_idx = set(info["sf_idx"])

        new_facts = set(info["last_action"]).intersection(self.sf_idx) - self.found_facts
        reward = len(new_facts) * self.per_fact_reward
        self.found_facts.update(new_facts)

        if not self.completed and self.found_facts == self.sf_idx:
            reward += self.completion_reward
            self.completed = True

        return reward
