import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ReplayBuffer
from utils.experiment import soft_update

from methods.networks import Actor, Critic


class DDPGAgent(nn.Module):
    def __init__(self, args, envs):
        super(DDPGAgent, self).__init__()

        # General Hyperparameters
        self.envs = envs
        self.args = args
        self.training = False
        self.obs_size = int(np.array(envs.single_observation_space.shape).prod())
        self.action_size = int(np.array(envs.single_action_space.shape).prod())
        self.obs_critic_size = self.obs_size + self.action_size
        self.replay_size = args.replay_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr_actor = args.learning_rate
        self.lr_critic = args.learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.replay_buffer = ReplayBuffer(self.replay_size, device=self.device)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(self.obs_critic_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.critic_loss_func = torch.nn.MSELoss()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(self.obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
        )

        # Optimizers
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=args.learning_rate
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.learning_rate
        )

        # Target Networks
        self.critic_target = nn.Sequential(
            nn.Linear(self.obs_critic_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.actor_target = nn.Sequential(
            nn.Linear(self.obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
        )

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.to(self.device)

    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def act(self, state, noisy=True):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        return action

    def train(self):
        train_logs = {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # ---------------------------- update critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        next_Qs = self.critic_target(torch.cat((next_states, next_actions), dim=1))

        # Compute Q targets for current states
        next_Qs[dones == 1.0] = 0.0
        Q_targets = rewards + self.gamma * next_Qs.detach()

        # Compute critic loss
        actual_Qs = self.critic(torch.cat((states, actions), dim=1))
        critic_loss = self.critic_loss_func(actual_Qs, Q_targets)

        # Minimize the loss
        critic_loss.backward()
        self.critic_optimizer.step()
        train_logs["loss_Q"] = critic_loss.item()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optimizer.zero_grad()
        actions_pred = self.actor(states)
        critic_evaluation = -self.critic(torch.cat((next_states, actions_pred), dim=1))
        actor_loss = critic_evaluation.mean()

        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()
        train_logs["loss_pi"] = actor_loss.item()

        # ----------------------- update target networks ----------------------- #
        soft_update(self.actor_target, self.actor, self.args.tau)
        soft_update(self.critic_target, self.critic, self.args.tau)

        return train_logs
