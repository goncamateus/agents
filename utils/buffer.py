import numpy as np
import random

import torch


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
            self.ptr = (self.ptr + 1) % self.max_size

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size, device="cpu"):
        """From a batch of experience, return values in Tensor form on device"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states_v = torch.Tensor(states).to(device)
        actions_v = torch.Tensor(actions).to(device)
        rewards_v = torch.Tensor(rewards).to(device)
        last_states_v = torch.Tensor(next_states).to(device)
        dones_t = torch.BoolTensor(dones).to(device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t

    def __len__(self):
        return len(self.buffer)
