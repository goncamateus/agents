import torch

from agents.torch.architectures.double_q import DoubleQNet


def test_double_q_net_forward():
    STATE_DIM = 4
    ACTION_DIM = 2
    HIDDEN_DIM = 256
    BACK_BONE_SIZE = 2

    model = DoubleQNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM, BACK_BONE_SIZE)
    state = torch.randn(1, STATE_DIM)
    action = torch.randn(1, ACTION_DIM)
    q1, q2 = model(state, action)

    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)
    assert not torch.equal(q1, q2)
