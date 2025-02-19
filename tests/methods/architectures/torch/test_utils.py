import pytest
import torch
import torch.nn as nn

from agents.methods.architectures.torch.utils import target_soft_update


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)


@pytest.fixture
def source_model():
    model = SimpleModel()
    for param in model.parameters():
        param.data.fill_(1.0)
    return model


@pytest.fixture
def target_model():
    model = SimpleModel()
    for param in model.parameters():
        param.data.fill_(0.0)
    return model


def test_target_soft_update(source_model, target_model):
    TAU = 0.5
    updated_target = target_soft_update(target_model, source_model, TAU)

    for target_param, source_param in zip(
        updated_target.parameters(), source_model.parameters()
    ):
        assert torch.allclose(target_param.data, source_param.data * TAU)


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_target_soft_update_various_tau(source_model, target_model, tau):
    updated_target = target_soft_update(target_model, source_model, tau)

    for target_param, source_param in zip(
        updated_target.parameters(), source_model.parameters()
    ):
        expected_value = source_param.data * tau
        assert torch.allclose(target_param.data, expected_value)
