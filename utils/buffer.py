import numpy as np
import random

import torch


class ReplayBuffer:
    def __init__(self, max_size, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        for i in range(len(state)):
            experience = (
                state[i],
                action[i],
                np.array(reward[i]),
                next_state[i],
                done[i],
            )
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
                self.ptr = (self.ptr + 1) % self.max_size

    def clear(self):
        self.buffer.clear()
        self.ptr = 0

    def sample(self, batch_size):
        """From a batch of experience, return values in Tensor form on device"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states_v = torch.Tensor(states).to(self.device)
        actions_v = torch.Tensor(actions).to(self.device)
        rewards_v = torch.Tensor(rewards).to(self.device)
        last_states_v = torch.Tensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t

    def __len__(self):
        return len(self.buffer)
