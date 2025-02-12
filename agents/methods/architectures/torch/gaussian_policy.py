import torch
import torch.nn as nn

from torch.distributions import Normal

from agents.methods.architectures.torch.utils import xavier_init


class GaussianPolicy(nn.Module):
    """
    A Gaussian Policy Network for reinforcement learning.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int, optional): Number of units in the hidden layers. Default is 256.
        back_bone_size (int, optional): Number of hidden layers in the backbone network. Default is 2.
        log_sig_min (float, optional): Minimum value for the log standard deviation. Default is -5.
        log_sig_max (float, optional): Maximum value for the log standard deviation. Default is 2.
        epsilon (float, optional): Small value to prevent division by zero in log probability calculation. Default is 1e-6.
        action_range (tuple, optional): Range of the action space (high, low). Default is (1, 0).

    Methods:
        forward(state):
            Forward pass through the network to compute the mean and log standard deviation of the action distribution.

        sample(state):
            Samples an action from the Gaussian policy using the reparameterization trick.
            Returns the action, log probability of the action, and the mean of the action distribution.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim: int = 256,
        back_bone_size: int = 2,
        log_sig_min=-5,
        log_sig_max=2,
        epsilon=1e-6,
        action_range=(torch.ones(1), torch.zeros(1)),
    ):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon

        self.back_bone_size = back_bone_size
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.back_bone = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(back_bone_size)
        ]
        self.back_bone = nn.Sequential(*self.back_bone)

        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.apply(xavier_init)

        self.action_scale = torch.FloatTensor((action_range[0] - action_range[1]) / 2.0)
        self.action_bias = torch.FloatTensor((action_range[0] + action_range[1]) / 2.0)

    def forward(self, state):
        x = torch.relu(self.input_layer(state))
        for layer in self.back_bone:
            x = torch.relu(layer(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
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
