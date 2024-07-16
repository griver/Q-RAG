import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = [None,]*max_size
        self.action = [None,]*max_size
        self.next_state = [None,]*max_size
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_episode(self, s, a, next_s, r, dones):
        correct_termination = (dones[-1] == True) and (not any(dones[:-1]))
        if not correct_termination:
            raise ValueError(f'strange episode with {sum(dones)} done values set to True')

        for s_i, a_i, next_s_i, r_i, done_i in zip(s,a, next_s, r, dones):
            self.add(s_i, a_i, next_s_i, r_i, done_i)


    def __len__(self):
        return self.size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        s = [self.state[i] for i in ind]
        a = [self.action[i] for i in ind]
        next_s = [self.next_state[i] for i in ind]
        r = self.reward[ind]
        not_done = self.not_done[ind]
        return s, a, r, next_s, not_done

    # def normalize_states(self, eps=1e-3):
    #     mean = self.state.mean(0, keepdims=True)
    #     std = self.state.std(0, keepdims=True) + eps
    #     self.state = (self.state - mean)/std
    #     self.next_state = (self.next_state - mean)/std
    #     return mean, std