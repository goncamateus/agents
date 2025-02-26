import torch

from agents.torch.architectures.q_net import QNet
from agents.torch.architectures.utils import xavier_init


def test_forward():
    my_q = QNet(10, 5, 3)
    observation = torch.randn(1, 10)
    action = torch.randn(1, 5)
    output = my_q(observation, action)
    assert output.shape == (1, 1)


def test_forward_with_initializer():
    my_q = QNet(10, 5, 3, initializer=xavier_init)
    observation = torch.randn(1, 10)
    action = torch.randn(1, 5)
    output = my_q(observation, action)
    assert output.shape == (1, 1)


def test_forward_with_multiple_outputs():
    my_q = QNet(10, 5, 3, number_outputs=2)
    observation = torch.randn(1, 10)
    action = torch.randn(1, 5)
    output = my_q(observation, action)
    assert output.shape == (1, 2)
