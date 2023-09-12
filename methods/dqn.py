import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.buffer import ReplayBuffer
from utils.experiment import StratLastRewards


class DQNAgent(nn.Module):
    def __init__(
        self,
        hyper_params,
        observation_space,
        action_space,
    ):
        super(DQNAgent, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.alpha = hyper_params.learning_rate
        self.gamma = hyper_params.gamma
        self.num_rewards = hyper_params.num_rewards
        self.epsilon = hyper_params.epsilon
        self.epsilon_end = hyper_params.epsilon_end
        self.epsilon_decay = hyper_params.epsilon_decay
        self.target_update_freq = hyper_params.target_update_freq
        self.n_epochs = hyper_params.num_epochs
        self.reward_scaling = hyper_params.reward_scaling
        self.n_updates = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and hyper_params.cuda else "cpu"
        )
        self.replay_buffer = ReplayBuffer(hyper_params.buffer_size, self.device)
        self.set_net()

        self.drQ = not hyper_params.dylam
        if self.drQ:
            self.lambdas = np.ones(self.num_rewards)
        else:
            self.lambdas = np.array(hyper_params.lambdas, dtype=np.float32)
        self.rew_tau = hyper_params.rew_tau
        self.rew_max = np.array(hyper_params.r_max, dtype=np.float32)
        self.rew_min = np.array(hyper_params.r_min, dtype=np.float32)
        self.stratified = hyper_params.stratified
        self.rb_rewards = StratLastRewards(hyper_params.episodes_rb, self.num_rewards)
        self.last_rew_mean = None

    def set_net(self):
        obs_size = (
            self.observation_space.n
            if isinstance(self.observation_space, gym.spaces.Discrete)
            else self.observation_space.shape[0]
        )

        def my_model():
            return nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_space.n),
            ).to(self.device)

        self.dqn = my_model()
        self.target_dqn = my_model()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.alpha)
        self.comp_dqns = [my_model() for _ in range(self.num_rewards)]
        self.comp_target_dqns = [my_model() for _ in range(self.num_rewards)]
        self.comp_optimizers = []
        for i in range(self.num_rewards):
            self.comp_target_dqns[i].load_state_dict(self.comp_dqns[i].state_dict())
            self.comp_optimizers.append(
                torch.optim.Adam(self.comp_dqns[i].parameters(), lr=self.alpha)
            )

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            q_values = self.dqn(torch.FloatTensor(state).to(self.device))
            action = q_values.argmax().item()
        else:
            action = self.action_space.sample()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action

    def store_reward(self, rewards):
        self.rb_rewards.add(rewards)

    def dylam(self):
        if self.stratified and self.rb_rewards.can_do() and not self.drQ:
            rew_mean_t = self.rb_rewards.mean()
            if self.last_rew_mean is not None:
                rew_mean_t = (
                    rew_mean_t + (self.last_rew_mean - rew_mean_t) * self.rew_tau
                )
            dQ = np.clip(
                (self.rew_max - rew_mean_t) / (self.rew_max - self.rew_min), 0, 1
            )
            expdQ = np.exp(dQ) - 1
            self.lambdas = expdQ / (np.sum(expdQ, 0) + 1e-4)
            self.last_rew_mean = rew_mean_t

    def update_components(self, batch):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = batch
        reward_batch = reward_batch * self.reward_scaling
        action_batch = action_batch.unsqueeze(1).long()
        if self.num_rewards == 1:
            reward_batch = reward_batch.unsqueeze(1)
        self.n_updates += 1
        losses = []
        if self.drQ:
            next_q_values = 0
            for i in range(self.num_rewards):
                with torch.no_grad():
                    next_q_values += self.comp_target_dqns[i](next_state_batch)
            next_q_values = next_q_values.max(1).values.unsqueeze(1)
            next_q_values[done_batch] = 0.0
        for i in range(self.num_rewards):
            if not self.drQ:
                with torch.no_grad():
                    next_q_values = (
                        self.comp_target_dqns[i](next_state_batch)
                        .max(1)
                        .values.unsqueeze(1)
                    )
                    next_q_values[done_batch] = 0.0
            y = reward_batch[:, i].unsqueeze(1) + self.gamma * next_q_values
            q_values = self.comp_dqns[i](state_batch)
            q_values = q_values.gather(1, action_batch)
            loss = F.mse_loss(q_values, y)
            self.comp_optimizers[i].zero_grad()
            loss.backward()
            losses.append(loss.item())
            self.comp_optimizers[i].step()
            if self.n_updates % self.target_update_freq == 0:
                self.comp_target_dqns[i].load_state_dict(self.comp_dqns[i].state_dict())
        return losses

    def update(self, batch_size):
        comp_losses = []
        losses = []
        for _ in range(self.n_epochs):
            batch = self.replay_buffer.sample(batch_size)
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
            ) = batch
            reward_batch = reward_batch * self.reward_scaling
            action_batch = action_batch.unsqueeze(1).long()
            if self.stratified:
                comp_loss = self.update_components(batch)
                comp_losses.append(comp_loss)
                Qs = torch.zeros((batch_size, self.num_rewards)).to(self.device)
                with torch.no_grad():
                    for i in range(self.num_rewards):
                        Qs[:, i] = (
                            self.comp_dqns[i](state_batch)
                            .gather(1, action_batch)
                            .squeeze()
                        )
                    lambs = torch.FloatTensor(self.lambdas).to(self.device)
                    Qs = torch.sum(lambs * Qs, 1).unsqueeze(1)
                actual_Qs = self.dqn(state_batch).gather(1, action_batch)
                loss = F.mse_loss(actual_Qs, Qs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            else:
                reward_batch = reward_batch.unsqueeze(1)
                with torch.no_grad():
                    next_q_values = (
                        self.target_dqn(next_state_batch).max(1).values.unsqueeze(1)
                    )
                    next_q_values[done_batch] = 0.0
                y = reward_batch + self.gamma * next_q_values
                q_values = self.dqn(state_batch)
                q_values = q_values.gather(1, action_batch)
                loss = F.mse_loss(q_values, y)
                self.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                if self.n_updates % self.target_update_freq == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
        mean_losses = np.mean(losses)
        mean_comp_losses = np.mean(comp_losses, axis=0) if comp_losses else None
        return mean_losses, mean_comp_losses

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.dqn.state_dict(), path + "dqn.pt")
        for i in range(self.num_rewards):
            torch.save(self.comp_dqns[i].state_dict(), path + f"comp_dqn_{i}.pt")

    def load(self, path):
        self.dqn.load_state_dict(torch.load(path + "dqn.pt"))
        self.eval()
        for i in range(self.num_rewards):
            self.comp_dqns[i].load_state_dict(torch.load(path + f"comp_dqn_{i}.pt"))
            self.comp_dqns[i].eval()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        for i in range(self.num_rewards):
            self.comp_target_dqns[i].load_state_dict(self.comp_dqns[i].state_dict())
