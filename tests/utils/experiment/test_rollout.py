import pytest
import numpy as np

from gonca_rl.agents import Agent
from gonca_rl.utils.experiment import Logger
from gonca_rl.utils.experiment.rollout import (
    end_of_episode,
    get_logs_from_info,
    get_log_from_reward,
    get_train_logs,
    update_agent,
    episode_rollout,
    update_logger,
    train_rollout,
)

import gymnasium as gym

from unittest.mock import Mock


@pytest.mark.parametrize(
    "done, truncated, expected",
    [
        pytest.param(True, False, True, id="TC1"),
        pytest.param(False, True, True, id="TC2"),
        pytest.param(False, False, False, id="TC3"),
    ],
)
def test_end_of_episode(done, truncated, expected):
    assert end_of_episode(done, truncated) is expected


@pytest.mark.parametrize(
    "info, info_keys, is_vector_env, expected",
    [
        pytest.param(
            {"key1": 1, "key2": 2},
            ["key1", "key2"],
            False,
            {"Episode/key1": 1, "Episode/key2": 2},
            id="TC1",
        ),
        pytest.param(
            {"key1": np.array([1, 2, 3]), "key2": np.array([4, 5, 6])},
            ["key1", "key2"],
            True,
            {"Episode/key1": 2.0, "Episode/key2": 5.0},
            id="TC2",
        ),
    ],
)
def test_get_logs_from_info(info, info_keys, is_vector_env, expected):
    logs = get_logs_from_info(info, info_keys, is_vector_env)
    assert logs == expected


@pytest.mark.parametrize(
    "reward, is_vector_env, expected",
    [
        pytest.param(10, False, {"Episode/Return": 10}, id="TC1"),
        pytest.param(np.array([1, 2, 3]), True, {"Episode/Return": 2.0}, id="TC2"),
    ],
)
def test_get_log_from_reward(reward, is_vector_env, expected):
    log = get_log_from_reward(reward, is_vector_env)
    assert log == expected


def test_get_train_logs():
    update_log = {"loss": 0.1, "loss1": 0.9}
    logs = get_train_logs(update_log)
    assert logs == {"Train/loss": 0.1, "Train/loss1": 0.9}


@pytest.fixture
def agent():
    agent = Mock(spec=Agent)
    agent.update_interval.return_value = 10
    agent.update.return_value = {"loss": 0.1}
    agent.get_action.return_value = 0
    return agent


@pytest.mark.parametrize(
    "global_step, expected",
    [
        pytest.param(1, None, id="TC1"),
        pytest.param(10, {"Train/loss": 0.1}, id="TC2"),
        pytest.param(15, None, id="TC3"),
        pytest.param(20, {"Train/loss": 0.1}, id="TC4"),
    ],
)
def test_update_agent(agent, global_step, expected):
    logs = update_agent(agent, global_step)
    assert logs == expected


@pytest.fixture
def env():
    env = Mock(spec=gym.Env)
    env.reset.return_value = (np.ones(2), {})
    return env


@pytest.mark.parametrize(
    "done, truncated, info_keys, is_training, expected",
    [
        pytest.param(
            True,
            False,
            [],
            True,
            ({"Episode/Return": 1}, [({"Train/loss": 0.1}, 0)]),
            id="TC1",
        ),
        pytest.param(
            False,
            True,
            [],
            True,
            ({"Episode/Return": 1}, [({"Train/loss": 0.1}, 0)]),
            id="TC2",
        ),
        # Test case 3
        pytest.param(True, False, [], False, ({"Episode/Return": 1}, []), id="TC3"),
        # Test case 4
        pytest.param(False, True, [], False, ({"Episode/Return": 1}, []), id="TC4"),
        # Test case 5
        pytest.param(
            True,
            False,
            ["key2"],
            True,
            ({"Episode/Return": 1, "Episode/key2": 2}, [({"Train/loss": 0.1}, 0)]),
            id="TC5",
        ),
        # Test case 6
        pytest.param(
            False,
            True,
            ["key2"],
            True,
            ({"Episode/Return": 1, "Episode/key2": 2}, [({"Train/loss": 0.1}, 0)]),
            id="TC6",
        ),
        # Test case 7 (this case should xfail)
        pytest.param(
            False,
            False,
            [],
            False,
            (),
            marks=pytest.mark.xfail(reason="Never ending episode"),
            id="TC7",
        ),
    ],
)
@pytest.mark.timeout(0.1)
def test_episode_rollout_using_one_environment(
    env, agent, done, truncated, info_keys, is_training, expected
):
    info = {
        "key1": 1,
        "key2": 2,
        "key3": 3,
    }
    env.step.return_value = (np.ones(2), 1, done, truncated, info)
    logs = episode_rollout(env, agent, 0, info_keys, is_training)
    assert logs == expected


@pytest.fixture
def vec_env():
    env = Mock(spec=gym.vector.VectorEnv)
    env.reset.return_value = (np.ones((4, 2)), {})
    return env


@pytest.mark.parametrize(
    "done, truncated, info_keys, is_training, expected",
    [
        pytest.param(
            np.array([False, False, False, False]),
            np.array([False, False, False, False]),
            [],
            True,
            (),
            id="TC1",
            marks=pytest.mark.xfail(reason="Never ending episode"),
        ),
        pytest.param(
            np.array([True, False, False, False]),
            np.array([False, False, True, False]),
            [],
            True,
            (),
            id="TC2",
            marks=pytest.mark.xfail(reason="Never ending episode"),
        ),
        pytest.param(
            np.array([False, True, False, False]),
            np.array([False, False, False, False]),
            [],
            True,
            (),
            id="TC3",
            marks=pytest.mark.xfail(reason="Never ending episode"),
        ),
        pytest.param(
            np.array([False, False, False, False]),
            np.array([False, False, True, False]),
            [],
            True,
            (),
            id="TC4",
            marks=pytest.mark.xfail(reason="Never ending episode"),
        ),
        pytest.param(
            np.array([True, True, True, True]),
            np.array([True, True, True, True]),
            [],
            True,
            ({"Episode/Return": 1}, [({"Train/loss": 0.1}, 0)]),
            id="TC5",
        ),
        pytest.param(
            np.array([True, True, True, True]),
            np.array([True, True, True, True]),
            ["key2"],
            True,
            ({"Episode/Return": 1, "Episode/key2": 3}, [({"Train/loss": 0.1}, 0)]),
            id="TC6",
        ),
    ],
)
@pytest.mark.timeout(0.1)
def test_episode_rollout_using_vectorized_environment(
    vec_env, agent, done, truncated, info_keys, is_training, expected
):
    info = {
        "key1": np.array([1, 2, 3, 4, 5]),
        "key2": np.array([1, 2, 3, 4, 5]),
        "key3": np.array([1, 2, 3, 4, 5]),
    }
    vec_env.step.return_value = (np.ones((4, 2)), np.ones(4), done, truncated, info)
    logs = episode_rollout(vec_env, agent, 0, info_keys, is_training)
    assert logs == expected


def test_update_logger():
    logger = Mock(spec=Logger)
    episode_logs = [{"Episode/Return": 1}]
    train_logs = [{"Train/loss": 0.1}]
    evaluation_logs = [{"Episode/Return": 1}]
    global_step = 0

    update_logger(logger, episode_logs, train_logs, evaluation_logs, global_step)
    logger.update_episode.assert_called_once_with(episode_logs, global_step)
    logger.update_train.assert_called_once_with(train_logs)
    logger.update_evaluation.assert_called_once_with(evaluation_logs, global_step)
    logger.push.assert_called_once()


# def test_train_rollout():
#     env = Mock(spec=gym.Env)
#     agent = Mock(spec=Agent)
#     logger = Mock(spec=Logger)
#     env.reset.return_value = "state"
#     env.step.return_value = ("next_state", 1, False, False, {"info_key": 1})
#     agent.get_action.return_value = "action"
#     agent.update_interval.return_value = 1
#     agent.update.return_value = {"loss": 0.1}

#     train_rollout(env, agent, logger, 10)
#     logger.update_episode.assert_called()
#     logger.update_train.assert_called()
#     logger.update_evaluation.assert_called()
#     logger.push.assert_called()
#     agent.save.assert_called()
#     env.close.assert_called_once()
#     logger.finish.assert_called_once()
