import pytest
import numpy as np

from agents.utils.replay_buffer.base import BaseReplayBuffer


@pytest.fixture
def replay_buffer():
    return BaseReplayBuffer(max_size=10)


@pytest.fixture
def experience():
    state = np.array([[1, 2, 3, 4, 5]], dtype=float)
    action = np.array([[6, 7, 8, 9, 10]], dtype=float)
    next_state = np.array([[11, 12, 13, 14, 15]], dtype=float)
    reward = np.array([16], dtype=float)
    done = np.array([False], dtype=bool)

    return state, action, reward, next_state, done


@pytest.fixture
def multiple_experiences():
    state = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float)
    action = np.array([[11], [12]], dtype=float)
    next_state = np.array([[13, 14, 15, 16, 17], [18, 19, 20, 21, 22]], dtype=float)
    reward = np.array([23, 24], dtype=float)
    done = np.array([False, True], dtype=bool)

    return state, action, reward, next_state, done


def test_add(replay_buffer, experience):
    replay_buffer.add(*experience)
    assert len(replay_buffer) == 1


def test_add_multiple_experiences(replay_buffer, multiple_experiences):
    replay_buffer.add(*multiple_experiences)
    assert len(replay_buffer) == 2


def test_clear(replay_buffer, multiple_experiences):
    while len(replay_buffer) < 10:
        replay_buffer.add(*multiple_experiences)
    replay_buffer.clear()
    assert len(replay_buffer) == 0
    assert replay_buffer.ptr == 0
