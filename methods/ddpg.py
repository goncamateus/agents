import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ReplayBuffer
from utils.experiment import soft_update
from utils.noise import OrnsteinUhlenbeckNoise as OUNoise

from methods.networks import Actor, Critic


class DDPGAgent(nn.Module):
    def __init__(self, args, envs):
        super(DDPGAgent, self).__init__()

        # General Hyperparameters
        self.envs = envs
        self.args = args
        self.training = False
        self.obs_size = np.array(envs.single_observation_space.shape).prod()
        self.action_size = np.array(envs.single_action_space.shape).prod()
        self.obs_critic_size = self.obs_size + self.action_size 
        self.replay_size = args.replay_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr_actor = args.learning_rate
        self.lr_critic = args.learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.replay_buffer = ReplayBuffer(self.replay_size)

        # DDPG specific
        self.noise_mu = args.noise_mu
        self.noise_sigma = args.noise_sigma
        self.noise_theta = args.noise_theta
        self.noise_epsilon_decay = 1 / args.epsilon_decay
        self.noise_epsilon = 1.0
        self.noise = OUNoise(
            self.noise_mu,
            self.noise_sigma,
            self.noise_theta,
            min_value=-1,
            max_value=1,
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(self.obs_critic_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic_loss_func = torch.nn.MSELoss()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_size),
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
            nn.Linear(self.obs_critic_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor_target = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_size),
            nn.Tanh(),
        )

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def act(self, state, noisy=True):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        if noisy:
            action += max(self.noise_epsilon, 0) * self.noise(action)
            action = np.clip(action, -1, 1)
            if self.training:
                self.noise_epsilon -= self.noise_epsilon_decay
        return action

    def train(self):
        train_logs = {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        next_Qs = self.critic_target(torch.cat((next_states, next_actions), dim=1))

        # Compute Q targets for current states
        next_Qs[dones] = 0.0
        Q_targets = rewards + self.gamma * next_Qs.detach()

        # Compute critic loss
        actual_Qs = self.critic(torch.cat((states, actions), dim=1))
        critic_loss = self.critic_loss_func(actual_Qs, Q_targets)
        train_logs["loss_Q"] = critic_loss.item()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        critic_evaluation = self.critic(torch.cat((next_states, actions_pred), dim=1))
        actor_loss = -critic_evaluation.mean()
        train_logs["loss_pi"] = actor_loss.item()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        soft_update(self.actor_target, self.actor, self.args.tau)
        soft_update(self.critic_target, self.critic, self.args.tau)

        return train_logs

