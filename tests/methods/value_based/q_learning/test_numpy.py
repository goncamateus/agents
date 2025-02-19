import pytest
import gymnasium as gym

from agents.methods.value_based.q_learning.numpy import NumpyQLearning as QLearning


@pytest.fixture
def sample_env():
    return gym.make("Taxi-v3")


@pytest.fixture
def agent(sample_env: gym.Env):
    return QLearning(
        {
            "gamma": 0.99,
            "alpha": 0.01,
            "epsilon": 1.0,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.01,
            "reward_scale": 1.0,
        },
        sample_env.observation_space,
        sample_env.action_space,
    )


def test_set_table(agent: QLearning):
    agent.set_table()
    assert agent.q_table.shape == (agent.num_states, agent.num_actions)


def test_get_output(agent: QLearning):
    agent.q_table = agent.q_table + 1
    observation = 0
    action = agent.get_output(observation)
    assert agent.action_space.contains(action)


def test_epsilon_greedy(agent: QLearning):
    agent.q_table = agent.q_table + 1
    observation = 0
    action = agent.epsilon_greedy(observation)
    assert isinstance(action, int)
    assert agent.action_space.contains(action)
