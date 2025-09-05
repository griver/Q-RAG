from abc import ABC, abstractmethod

class AFeedbackModel(ABC):
    FEEDBACK_MODEL_NAME: str

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

    @abstractmethod
    def copy(self):
        pass


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
    def __init__(self, penalize_extra_steps=False, completion_reward=1.0, per_fact_reward=0.0):
        super().__init__()
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
