import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.buffer import ReplayBuffer
from torch.distributions import Normal
from torch.optim import Adam

from utils.experiment import StratLastRewards


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_rewards=1, hidden_dim=256):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rewards),
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rewards),
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state.clone(), action.clone()], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        epsilon=1e-6,
        action_space=None,
    ):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_action = torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob - log_action
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, state):
        action, _, _ = self.sample(state)
        return action.detach().cpu().numpy()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class TargetCritic(TargetNet):
    def __call__(self, S, A):
        return self.target_model(S, A)


class SACStrat(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        original_weights,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
    ):

        super(SACStrat, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = np.array(observation_space.shape).prod()
        self.num_actions = np.array(action_space.shape).prod()
        self.num_rewards = args.num_rewards
        self.actor = GaussianPolicy(
            self.num_inputs,
            self.num_actions,
            log_sig_min=log_sig_min,
            log_sig_max=log_sig_max,
            hidden_dim=hidden_dim,
            epsilon=args.epsilon,
            action_space=action_space,
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.critic = QNetwork(
            self.num_inputs,
            self.num_actions,
            num_rewards=self.num_rewards,
            hidden_dim=hidden_dim,
        )
        self.critic_target = TargetCritic(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)

        # Automatic entropy tuning
        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optim = None
        if args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.device)

        # DyLam
        self.original_weights = torch.Tensor(original_weights).to(self.device)
        self.last_epi_rewards = StratLastRewards(args.episodes_rb, self.num_rewards)
        self.last_rew_mean = None
        self.to(self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.target_model.to(device)
        return super(SACStrat, self).to(device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor.get_action(state)

    def update(self, batch_size, lambdas, update_actor=False):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)
        
        # reward_batch = reward_batch*2000

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target.target_model(
                next_state_batch, next_state_action
            )

            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            min_qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target
            next_q_value = next_q_value.sum(1)
        
        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1 = qf1.sum(1)
        qf2 = qf2.sum(1)

        # qf1_loss = 0
        # qf2_loss = 0
        # for i in range(self.num_rewards):
        #     # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #     qf1_loss += F.mse_loss(qf1[:, i], next_q_value[:, i])
        #     # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #     qf2_loss +=  F.mse_loss(qf2[:, i], next_q_value[:, i])
        qf1_loss = F.mse_loss(qf1, next_q_value)

        qf2_loss = F.mse_loss(qf2, next_q_value)

        # Minimize the loss between two Q-functions
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        policy_loss = None
        alpha_loss = None
        if update_actor:
            pi, log_pi, _ = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            min_qf_pi = (min_qf_pi*lambdas).sum(1).view(-1, 1)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = self.alpha * log_pi
            import ipdb; ipdb.set_trace()
            policy_loss = policy_loss - min_qf_pi
            policy_loss = policy_loss.mean()
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            if self.alpha_optim is not None:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.sample(state_batch)
                alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
                alpha_loss = alpha_loss.mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().item()

        return policy_loss, qf1_loss, qf2_loss, alpha_loss
