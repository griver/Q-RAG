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
        self.next_action = [None,]*max_size
        self.q_values = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.entropy = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, next_action, reward, done, entropy, max_q):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.next_action[self.ptr] = next_action
        self.reward[self.ptr] = reward
        self.entropy[self.ptr] = entropy
        self.q_values[self.ptr] = max_q
        self.not_done[self.ptr] = 1. - int(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_episode(self, s, a, next_s, r, dones):
        correct_termination = (dones[-1] == True) and (not any(dones[:-1]))
        if not correct_termination:
            raise ValueError(f'strange episode with {sum(dones)} done values set to True')

        next_a = a[1:] + [a[-1]]
        for s_i, a_i, next_s_i, next_a_i, r_i, done_i in zip(s, a, next_s, next_a, r, dones):
            self.add(s_i, a_i, next_s_i, next_a_i, r_i, done_i, 0, 0)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        s = [self.state[i] for i in ind]
        a = [self.action[i] for i in ind]
        next_s = [self.next_state[i] for i in ind]
        next_a = [self.next_action[i] for i in ind]
        r = self.reward[ind]
        not_done = self.not_done[ind]
        entropy = self.entropy[ind]
        return s, a, r, next_s, next_a, not_done, entropy
    
    def ordered_sample(self, batch_size):
        start = np.random.randint(0, self.size - batch_size + 1)
        ind = np.arange(start, start + batch_size)

        s = [self.state[i] for i in ind]
        a = [self.action[i] for i in ind]
        next_s = [self.next_state[i] for i in ind]
        next_a = [self.next_action[i] for i in ind]
        r = self.reward[ind]
        q = self.q_values[ind]
        not_done = self.not_done[ind]
        entropy = self.entropy[ind]
        return s, a, r, next_s, next_a, not_done, entropy, q

