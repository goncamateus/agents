import numpy as np


class StratLastRewards:
    def __init__(self, size, n_rewards):
        self.pos = 0
        self.size = size
        self._can_do = False
        self.rewards = np.zeros((size, n_rewards))

    def add(self, reward):
        self.rewards[self.pos] = reward
        if self.pos == (self.size - 1):
            self._can_do = True
        self.pos = (self.pos + 1) % self.rewards.shape[0]

    def can_do(self):
        return self._can_do

    def mean(self):
        return self.rewards.mean(0)
