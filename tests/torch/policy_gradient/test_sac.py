import gymnasium as gym
import numpy as np
import pytest

from agents.torch.policy_gradient.sac import TorchSAC as SAC
from agents.torch.utils.replay_buffer import TorchReplayBuffer as ReplayBuffer


@pytest.fixture
def hyper_parameters():
    return {
        "hidden_dim": 256,
        "action_range": (-1, 1),
        "automatic_entropy_tuning": True,
        "q_learning_rate": 3e-4,
        "policy_learning_rate": 3e-4,
        "buffer_size": 1000000,
        "reward_scale": 1.0,
        "gamma": 0.99,
        "alpha": 0.2,
        "device": "cpu",
    }


@pytest.fixture(scope="session")
def env():
    vec_env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")
    vec_env.reset()
    yield vec_env
    vec_env.close()


@pytest.fixture
def observation_space(env):
    return env.single_observation_space


@pytest.fixture
def action_space(env):
    return env.single_action_space


@pytest.fixture
def sac_agent(hyper_parameters, observation_space, action_space):
    return SAC(hyper_parameters, observation_space, action_space)


@pytest.fixture(scope="session")
def agent_with_memory(env):
    agent = SAC(
        {
            "hidden_dim": 256,
            "action_range": (-1, 1),
            "automatic_entropy_tuning": True,
            "q_learning_rate": 3e-4,
            "policy_learning_rate": 3e-4,
            "buffer_size": 1000000,
            "reward_scale": 1.0,
            "gamma": 0.99,
            "alpha": 0.2,
            "device": "cpu",
        },
        env.single_observation_space,
        env.single_action_space,
    )
    for _ in range(100):
        action = env.action_space.sample()
        state = env.observation_space.sample()
        next_state = env.observation_space.sample()
        reward = np.array([10 for _ in range(10)])
        done = np.array([True for _ in range(10)])
        agent.replay_buffer.add(state, action, reward, next_state, done)
    return agent


def test_build_networks(sac_agent):
    sac_agent.build_networks()
    assert sac_agent.critic is not None
    assert sac_agent.actor is not None
    assert sac_agent.target_entropy is not None
    assert sac_agent.log_alpha is not None


def test_set_target_networks(sac_agent):
    sac_agent.set_target_networks()
    assert sac_agent.target_critic is not None
    assert sac_agent.target_actor is not None


def test_build_optimizers(sac_agent):
    sac_agent.build_optimizers()
    assert sac_agent.critic_optimizer is not None
    assert sac_agent.actor_optimizer is not None
    assert sac_agent.alpha_optimizer is not None


def test_set_device(sac_agent):
    sac_agent.build_networks()
    sac_agent.set_device()
    assert sac_agent.device.type == "cpu"


def test_init_replay_buffer(sac_agent):
    sac_agent.init_replay_buffer()
    assert isinstance(sac_agent.replay_buffer, ReplayBuffer)


def test_get_action(sac_agent, env):
    observation, _ = env.reset()
    action = sac_agent.get_action(observation)[0]
    assert action.shape == env.single_action_space.shape


def test_update_critic(agent_with_memory):
    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
    ) = agent_with_memory.replay_buffer.sample(4)
    qf1_loss, qf2_loss = agent_with_memory.update_critic(
        state_batch, action_batch, reward_batch, next_state_batch, done_batch
    )
    assert qf1_loss is not None
    assert qf2_loss is not None


def test_update_actor(agent_with_memory):
    state_batch, _, _, _, _ = agent_with_memory.replay_buffer.sample(4)
    policy_loss, alpha_loss = agent_with_memory.update_actor(state_batch)
    assert policy_loss is not None
    assert alpha_loss is not None


def test_update(agent_with_memory):
    policy_loss, qf1_loss, qf2_loss, alpha_loss = agent_with_memory.update(batch_size=4)
    assert policy_loss is not None
    assert qf1_loss is not None
    assert qf2_loss is not None
    assert alpha_loss is not None


def test_save_load(sac_agent, tmp_path):
    sac_agent.build_networks()
    sac_agent.set_target_networks()
    sac_agent.save(tmp_path)
    sac_agent.load(tmp_path)
    assert sac_agent.actor is not None
    assert sac_agent.critic is not None
