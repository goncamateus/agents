import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils.experiment import StratLastRewards
from utils.network import layer_init


class PPOStrat(nn.Module):
    def __init__(self, args, envs):
        super(PPOStrat, self).__init__()
        self.obs_size = np.array(envs.single_observation_space.shape)
        self.action_size = envs.single_action_space.n
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, args.num_rewards), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)
        self.last_epi_rewards = StratLastRewards(
            self.args.episodes_rb, self.args.num_rewards
        )
        self.last_rew_mean = None

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def update(self, clipfracs, batch, alphas):
        obs = batch[0]
        logprobs = batch[1]
        actions = batch[2]
        advantages = batch[3]
        returns = batch[4]
        values = batch[5]

        _, newlogprob, entropy, newvalue = self.get_action_and_value(
            obs, actions.long()
        )
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [
                ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            ]

        mb_advantages = advantages
        if self.args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        # Policy loss
        mb_advantages = (mb_advantages * alphas).sum(1)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -self.args.clip_coef,
                self.args.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
        self.optimizer.step()

        return old_approx_kl, approx_kl, v_loss, pg_loss, entropy_loss, alphas


class PPOStratContinuous(PPOStrat):
    def __init__(self, args, envs):
        super(PPOStratContinuous, self).__init__(args, envs)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_actor_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_size))
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
