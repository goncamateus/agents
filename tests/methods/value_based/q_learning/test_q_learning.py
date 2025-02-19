import pytest
import numpy as np
import gymnasium as gym

from agents.methods.value_based.q_learning.q_learning import QLearning


@pytest.fixture
def sample_env():
    return gym.make("Taxi-v3")


@pytest.fixture
def agent(sample_env: gym.Env, monkeypatch):
    QLearning.__abstractmethods__ = set()

    def set_table(self):
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def epsilon_greedy(self, observation):
        return 0

    monkeypatch.setattr(QLearning, "set_table", set_table)
    monkeypatch.setattr(QLearning, "epsilon_greedy", epsilon_greedy)

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


def test_hyper_parameters(agent: QLearning):
    agent.hyper_parameters(
        {
            "gamma": 0.98,
            "alpha": 0.02,
            "epsilon": 2.0,
            "epsilon_decay": 0.989,
            "epsilon_min": 0.02,
            "reward_scale": 120,
        }
    )
    assert agent.gamma == 0.98
    assert agent.alpha == 0.02
    assert agent.epsilon == 2.0
    assert agent.epsilon_decay == 0.989
    assert agent.epsilon_min == 0.02
    assert agent.reward_scale == 120


@pytest.mark.parametrize(
    "space, expected",
    [
        (gym.spaces.Discrete(10), ()),
        (gym.spaces.Box(-1, 1, (3, 3)), (3, 3)),
        pytest.param(gym.spaces.MultiDiscrete([2, 3, 4]), (), marks=pytest.mark.xfail),
        pytest.param(gym.spaces.MultiBinary(10), (), marks=pytest.mark.xfail),
        pytest.param(
            gym.spaces.Tuple([gym.spaces.Discrete(10), gym.spaces.Discrete(10)]),
            (),
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_set_input_space(agent: QLearning, space: gym.Space, expected: tuple):
    agent.set_input_space(space)
    assert agent.observation_space.shape == expected


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(gym.spaces.MultiDiscrete([2, 3, 4])),
        pytest.param(gym.spaces.MultiBinary(10)),
        pytest.param(
            gym.spaces.Tuple([gym.spaces.Discrete(10), gym.spaces.Discrete(10)])
        ),
    ],
)
def test_set_input_space_raise_error_when_wrong_space(
    agent: QLearning, space: gym.Space
):
    with pytest.raises(ValueError):
        agent.set_input_space(space)


@pytest.mark.parametrize(
    "space, expected",
    [
        (gym.spaces.Discrete(10), 10),
        pytest.param(gym.spaces.Box(-1, 1, (3, 3)), 9, marks=pytest.mark.xfail),
        pytest.param(gym.spaces.MultiDiscrete([2, 3, 4]), 9, marks=pytest.mark.xfail),
        pytest.param(gym.spaces.MultiBinary(10), 9, marks=pytest.mark.xfail),
        pytest.param(
            gym.spaces.Tuple([gym.spaces.Discrete(10), gym.spaces.Discrete(10)]),
            9,
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_set_output_space(agent: QLearning, space: gym.Space, expected: int):
    agent.set_output_space(space)
    assert agent.num_actions == expected


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(gym.spaces.MultiDiscrete([2, 3, 4])),
        pytest.param(gym.spaces.MultiBinary(10)),
        pytest.param(
            gym.spaces.Tuple([gym.spaces.Discrete(10), gym.spaces.Discrete(10)])
        ),
    ],
)
def test_set_output_space_raise_error_when_wrong_space(
    agent: QLearning, space: gym.Space
):
    with pytest.raises(ValueError):
        agent.set_output_space(space)


def test_get_action(agent: QLearning):
    action = agent.get_action(0)
    assert action < agent.num_actions
    assert action >= 0
    assert isinstance(action, int)


def test_update(agent: QLearning):
    table_before = agent.q_table.copy()
    agent.update(0, 0, 100, 1)
    assert (table_before != agent.q_table).any()


def test_save(agent: QLearning, tmp_path):
    agent.save(tmp_path / "test.pt")
    assert (tmp_path / "test.pt").exists()


def test_load(agent: QLearning, tmp_path):
    table_before = agent.q_table.copy()
    agent.save(tmp_path / "test.pt")
    agent.load(tmp_path / "test.pt")
    assert (agent.q_table == table_before).all()
