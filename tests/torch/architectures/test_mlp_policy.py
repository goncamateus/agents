import torch

from agents.torch.architectures.mlp_policy import MLPPolicy
from agents.torch.architectures.utils import xavier_init


def test_forward():
    my_policy = MLPPolicy(10, 5, 3)
    observation = torch.randn(1, 10)
    output = my_policy(observation)
    assert output.shape == (1, 5)


def test_forward_with_initializer():
    my_policy = MLPPolicy(10, 5, 3, initializer=xavier_init)
    observation = torch.randn(1, 10)
    output = my_policy(observation)
    assert output.shape == (1, 5)
