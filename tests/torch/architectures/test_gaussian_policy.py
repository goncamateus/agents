import pytest
import torch

from agents.torch.architectures.gaussian_policy import GaussianPolicy


@pytest.fixture
def gaussian_policy():
    state_dim = 4
    action_dim = 2
    return GaussianPolicy(state_dim, action_dim)


def test_gaussian_policy_forward(gaussian_policy):
    state = torch.randn(1, 4)
    mean, log_std = gaussian_policy.forward(state)
    assert mean.shape == (1, 2)
    assert log_std.shape == (1, 2)
    assert torch.all(log_std >= gaussian_policy.log_sig_min)
    assert torch.all(log_std <= gaussian_policy.log_sig_max)


def test_gaussian_policy_sample(gaussian_policy):
    state = torch.randn(1, 4)
    action, log_prob, mean = gaussian_policy.sample(state)
    assert action.shape == (1, 2)
    assert log_prob.shape == (1, 1)
    assert mean.shape == (1, 2)
