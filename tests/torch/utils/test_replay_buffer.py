import numpy as np
import pytest

from agents.torch.utils.replay_buffer import TorchReplayBuffer


@pytest.fixture
def replay_buffer():
    return TorchReplayBuffer(max_size=10)


@pytest.fixture
def multiple_experiences():
    state = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float)
    action = np.array([[11], [12]], dtype=float)
    next_state = np.array([[13, 14, 15, 16, 17], [18, 19, 20, 21, 22]], dtype=float)
    reward = np.array([23, 24], dtype=float)
    done = np.array([False, True], dtype=bool)

    return state, action, reward, next_state, done


@pytest.mark.parametrize("batch_size", [1, 2])
def test_sample(replay_buffer, multiple_experiences, batch_size):
    replay_buffer.add(*multiple_experiences)
    batch = replay_buffer.sample(batch_size)
    assert batch[0].shape[0] == batch_size
