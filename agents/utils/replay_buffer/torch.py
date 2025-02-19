import random
import numpy as np
import torch

from agents.utils.replay_buffer.base import BaseReplayBuffer


class TorchReplayBuffer(BaseReplayBuffer):
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states_v = torch.Tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_v = torch.Tensor(rewards).to(self.device)
        last_states_v = torch.Tensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t
