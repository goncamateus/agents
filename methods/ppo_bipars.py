import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.buffer import ReplayBuffer
from utils.experiment import HyperParameters
from utils.network import layer_init
from torch.distributions.categorical import Categorical
from methods.ppo import PPO


class PPOBiPaRS(nn.Module):
    def __init__(self, args, envs):
        super(PPOBiPaRS, self).__init__()
        self.obs_size = np.array(envs.single_observation_space.shape).prod()
        self.action_size = envs.single_action_space.n
        self.args = args
        self.ppo = PPO(args, envs)
        self.z = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.optimizer = optim.Adam(self.z.parameters(), lr=args.learning_rate)

    def get_value(self, x):
        return self.ppo.critic(x)

    def get_action_and_value(self, x, action=None):
        z = self.z(x)
        zcat = torch.cat([x, z], dim=1)
        logits = self.ppo.actor(zcat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.ppo.critic(x)

    def update(self, clipfracs, batch):
        obs = batch[0]
        logprobs = batch[1]
        actions = batch[2]

        # Update PPO Actor and Critic
        old_approx_kl, approx_kl, v_loss, pg_loss, entropy_loss = self.ppo.update(
            clipfracs, batch
        )

        # Compute the loss for the z-network
        _, logprobs, _, Qs = self.get_action_and_value(obs, actions.long())
        loss = -(logprobs * Qs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return old_approx_kl, approx_kl, v_loss, pg_loss, entropy_loss

