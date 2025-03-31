from abc import ABC, abstractmethod

class ARewardModel(ABC):

    @abstractmethod
    def reset(self, obs, info):
        pass

    @abstractmethod
    def reward(self, obs, info):
        pass


class GroundTruthReward(ARewardModel):

    def __init__(self, per_fact_reward=0.1, completion_reward=1.0):
        self.per_fact_reward = per_fact_reward
        self.completion_reward = completion_reward
        self.sf_idx = None
        self.found_facts = set()
        self.completed = False

    def reset(self, obs, info) -> None:
        self.sf_idx = None
        self.found_facts.clear()
        self.completed = False

    def reward(self, obs, info) -> float:
        if self.sf_idx is None:
            self.sf_idx = set(info["sf_idx"])

        new_facts = set(info["last_action"]).intersection(self.sf_idx) - self.found_facts
        reward = len(new_facts) * self.per_fact_reward
        self.found_facts.update(new_facts)

        if not self.completed and self.found_facts == self.sf_idx:
            reward += self.completion_reward
            self.completed = True

        return reward