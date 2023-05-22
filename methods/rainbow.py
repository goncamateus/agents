"""
    Rainbow DQN.
    Author: Jinwoo Park
    GitHub: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master
"""
import math
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.experiment import soft_update

from utils.per import PrioritizedReplayBuffer, ReplayBuffer


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.out_dim = out_dim

        # set common feature layer
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.net(x)


class RainbowDQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self,
        arguments,
        observation_space: gym.spaces,
        action_space: gym.spaces,
        tau=0.001,
    ):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.n

        self.batch_size = arguments.batch_size
        self.target_update = arguments.target_network_frequency
        self.gamma = arguments.gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device("cuda" if arguments.cuda else "cpu")

        self.tau = tau
        # PER
        # memory for 1-step Learning
        self.beta = arguments.beta
        self.prior_eps = arguments.prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, arguments.buffer_size, arguments.batch_size, alpha=arguments.alpha
        )

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=arguments.q_lr)

        # transition to store in memory
        self.transition = list()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def update(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 1)
        self.optimizer.step()

        soft_update(self.dqn, self.dqn_target, self.tau)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device).reshape(-1, 1)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        # Double DQN loss
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = (
            self.dqn_target(next_state)
            .gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True))
            .detach()
        )
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)
        elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")
        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
