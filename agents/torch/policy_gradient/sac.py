import pathlib
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
from torch.optim import Adam

from agents.common.policy_gradient.sac import SAC
from agents.torch.architectures.double_q import DoubleQNet
from agents.torch.architectures.gaussian_policy import GaussianPolicy
from agents.torch.utils.replay_buffer import TorchReplayBuffer as ReplayBuffer


class TorchSAC(SAC, nn.Module):
    def __init__(self, hyper_parameters, observation_space, action_space):
        nn.Module.__init__(self)
        SAC.__init__(self, hyper_parameters, observation_space, action_space)
        self.device = torch.device(self.device_name)
        self.to(self.device)

    def to(self, device):
        self.device = device
        self.replay_buffer.device = self.device
        self.actor.action_bias = self.actor.action_bias.to(self.device)
        self.actor.action_scale = self.actor.action_scale.to(self.device)
        self.log_alpha = self.log_alpha.to(self.device)
        return super().to(device)

    def build_networks(self):
        self.critic = DoubleQNet(
            self.num_inputs,
            self.num_outpts,
            self.hidden_dim,
        )
        self.actor = GaussianPolicy(
            self.num_inputs,
            self.num_outpts,
            hidden_dim=self.hidden_dim,
            log_sig_min=self.log_sig_min,
            log_sig_max=self.log_sig_max,
            epsilon=self.epsilon,
            action_range=self.action_range,
        )
        self.target_entropy = None
        self.log_alpha = None
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_range[0].shape)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()

    def set_target_networks(self):
        self.target_critic = deepcopy(self.critic)

    def build_optimizers(self):
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.q_learning_rate)
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.policy_learning_rate
        )
        if self.automatic_entropy_tuning:
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.policy_learning_rate)

    def init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def get_action(self, observations: np.ndarray, deterministic=False):
        observations = torch.FloatTensor(observations).to(self.device)
        action, _, deterministic_action = self.actor.sample(observations)
        action = deterministic_action if deterministic else action
        return action.cpu().detach().numpy()

    def update_critic(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.FloatTensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        done_batch: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.target_critic(
                next_state_batch, next_state_action
            )
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
            min_qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target
        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)

        # Minimize the loss between two Q-functions
        qf_loss = qf1_loss + qf2_loss
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        return qf1_loss, qf2_loss

    def update_alpha(self, state_batch: torch.Tensor) -> torch.Tensor:
        alpha_loss = None
        if self.alpha_optimizer is not None:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state_batch)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
            alpha_loss = alpha_loss.mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        return alpha_loss

    def update_actor(
        self, state_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = self.update_alpha(state_batch)

        return policy_loss, alpha_loss

    def update(self, batch_size, update_actor=True):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)

        reward_batch = reward_batch * self.reward_scale
        qf1_loss, qf2_loss = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch)

        return policy_loss, qf1_loss, qf2_loss, alpha_loss

    def save(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")

    def load(self, path):
        path = pathlib.Path(path)
        self.actor.load_state_dict(
            torch.load(path / "actor.pt", map_location=self.device, weights_only=True),
        )
        self.critic.load_state_dict(
            torch.load(path / "critic.pt", map_location=self.device, weights_only=True),
        )
        self.actor.eval()
        self.critic.eval()
