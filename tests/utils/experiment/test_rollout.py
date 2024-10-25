import pytest
import numpy as np

from gonca_rl.agents import Agent
from gonca_rl.utils.experiment import Logger
from gonca_rl.utils.experiment.rollout import (
    update_agent,
    treats_next_state,
    evaluate,
    rollout,
)

import gymnasium as gym

from unittest.mock import Mock


@pytest.fixture
def logger():
    def patched_update_episode(
        episode_reward, info, done, truncated, global_step, info_keys
    ):
        return None

    def patched_update_train(train_logs, global_step):
        return None

    _logger = Mock(spec=Logger)
    _logger.update_episode.side_effect = patched_update_episode
    _logger.update_train.side_effect = patched_update_train
    return _logger


@pytest.fixture
def agent():
    _agent = Mock(spec=Agent)
    _agent.update_interval.return_value = 10
    _agent.update.return_value = {"loss": 0.1}
    _agent.get_action.return_value = 0
    return _agent


@pytest.fixture
def env():
    def patched_step(action):
        done = False if np.random.rand() > 0.5 else True
        truncated = False if np.random.rand() > 0.5 else True
        return (np.ones(2), 1, done, truncated, {"key1": 1, "key2": 2, "key3": 3})

    env = Mock(spec=gym.Env)
    env.reset.return_value = (np.ones(2), {})
    env.step.side_effect = patched_step
    return env


@pytest.fixture
def vec_env():
    def patched_step(action):
        done = np.random.rand(4) > 0.5
        truncated = np.random.rand(4) > 0.5
        return (
            np.ones((4, 2)),
            np.ones(4),
            done,
            truncated,
            {"key1": np.ones(4), "key2": 2 * np.ones(4), "key3": 3 * np.ones(4)},
        )

    env = Mock(spec=gym.vector.VectorEnv)
    env.reset.return_value = (np.ones((4, 2)), {})
    env.step.side_effect = patched_step
    return env


@pytest.mark.parametrize(
    "global_step, expected",
    [
        pytest.param(1, 0, id="TC1"),
        pytest.param(10, 1, id="TC2"),
        pytest.param(15, 0, id="TC3"),
        pytest.param(20, 1, id="TC4"),
    ],
)
def test_update_agent(agent, logger, global_step, expected):
    update_agent(agent, global_step, logger)
    assert agent.update.call_count == expected
    assert logger.update_train.call_count == expected


@pytest.mark.parametrize(
    "state, next_state, done, is_vector_env, expected",
    [
        pytest.param(
            np.ones(2), 2 * np.ones(2), False, False, 2 * np.ones(2), id="TC1"
        ),
        pytest.param(np.ones(2), 2 * np.ones(2), True, False, np.ones(2), id="TC2"),
        pytest.param(
            np.ones((2, 2)),
            2 * np.ones((2, 2)),
            np.array([False, False]),
            True,
            2 * np.ones((2, 2)),
            id="TC1-VecEnv",
        ),
        pytest.param(
            np.ones((2, 2)),
            2 * np.ones((2, 2)),
            np.array([True, False]),
            True,
            np.array([[1, 1], [2, 2]], dtype=float),
            id="TC2-VecEnv",
        ),
        pytest.param(
            np.ones((2, 2)),
            2 * np.ones((2, 2)),
            np.array([False, True]),
            True,
            np.array([[2, 2], [1, 1]], dtype=float),
            id="TC3-VecEnv",
        ),
    ],
)
def test_treats_next_state(state, next_state, done, is_vector_env, expected):
    result = treats_next_state(state, next_state, done, is_vector_env)
    assert np.all(result == expected)
