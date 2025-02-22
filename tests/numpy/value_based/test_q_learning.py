import gymnasium as gym
import numpy as np
import pytest

from agents.numpy.value_based.q_learning import NumpyQLearning as QLearning


@pytest.fixture
def sample_env() -> gym.Env:
    return gym.make("Taxi-v3")


@pytest.fixture
def hyper_parameters() -> dict:
    return {
        "gamma": 0.99,
        "alpha": 0.1,
        "epsilon": 0.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "reward_scale": 1.0,
    }


@pytest.fixture
def q_agent(hyper_parameters: dict, sample_env: gym.Env) -> QLearning:
    agent = QLearning(
        hyper_parameters=hyper_parameters,
        observation_space=sample_env.observation_space,
        action_space=sample_env.action_space,
    )
    return agent


def test_set_table(q_agent: QLearning):
    q_agent.set_table()
    assert np.all(q_agent.q_table == 0)


def test_get_output_equal_row(q_agent: QLearning):
    action = q_agent.get_output(0)
    assert action < q_agent.num_actions
    assert action >= 0
    assert isinstance(action, int)


def test_get_output_diff_row(q_agent: QLearning):
    row = np.zeros(q_agent.num_actions)
    row[0] = 1
    q_agent.q_table[0] = row
    action = q_agent.get_output(0)
    assert action == 0


def test_epsilon_greedy_greedy_action(
    q_agent: QLearning, monkeypatch: pytest.MonkeyPatch
):
    def mock_random():
        return 0

    monkeypatch.setattr(np.random, "random", mock_random)
    row = np.zeros(q_agent.num_actions)
    row[0] = 1
    q_agent.q_table[0] = row
    action = q_agent.epsilon_greedy(0)
    assert action == 0


def test_epsilon_greedy_random_action(
    q_agent: QLearning, monkeypatch: pytest.MonkeyPatch
):
    def mock_random():
        return 1

    monkeypatch.setattr(np.random, "random", mock_random)
    action = q_agent.epsilon_greedy(0)
    assert action < q_agent.num_actions
    assert action >= 0
    assert isinstance(action, int)
