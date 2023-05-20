import dataclasses
import os

import gym
import numpy as np
# import rsoccer_gym
import torch

from utils.async_vec import AsyncVectorEnv
from utils.wrappers import RecordEpisodeStatistics


class StratLastRewards:
    def __init__(self, size, n_rewards):
        self.pos = 0
        self.size = size
        self._can_do = False
        self.rewards = np.zeros((size, n_rewards))

    def add(self, rewards):
        self.rewards[self.pos] = rewards
        if self.pos == self.size - 1:
            self._can_do = True
        self.pos = (self.pos + 1) % self.rewards.shape[0]

    def can_do(self):
        return self._can_do

    def mean(self):
        return self.rewards.mean(0)


@dataclasses.dataclass
class HyperParameters:
    """Class containing all experiment hyperparameters"""

    EXP_NAME: str
    ENV_NAME: str
    LEARNING_RATE: float
    REPLAY_SIZE: int  # Maximum Replay Buffer Sizer
    REPLAY_INITIAL: int  # Minimum experience buffer size to start training
    SAVE_FREQUENCY: int  # Save checkpoint every _ grad_steps
    BATCH_SIZE: int
    GAMMA: float  # Reward Decay
    MAX_EPISODE_STEPS: int = None
    N_OBS: int = None
    N_ACTS: int = None
    SAVE_PATH: str = None
    DEVICE: str = None
    TOTAL_GRAD_STEPS: int = None
    NOISE_SIGMA_INITIAL: float = None
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: int = None

    def to_dict(self):
        return self.__dict__

    def __post_init__(self):
        env = gym.make(self.ENV_NAME)
        self.N_OBS, self.N_ACTS = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.SAVE_PATH = os.path.join("saves", self.ENV_NAME, self.EXP_NAME)
        self.CHECKPOINT_PATH = os.path.join(self.SAVE_PATH, "checkpoints")
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        self.action_space = env.action_space
        self.observation_space = env.observation_space


def save_checkpoint(hp, steps, neural_nets, optimizers):
    checkpoint = dataclasses.asdict(hp)
    for name, net in neural_nets.items():
        checkpoint[name] = net.state_dict()
    for name, optim in optimizers.items():
        checkpoint[name] = optim.state_dict()
    filename = os.path.join(hp.CHECKPOINT_PATH, "checkpoint_{:09}.pth".format(steps))
    torch.save(checkpoint, filename)


def soft_update(model, target, tau):
    """
    Blend params of target net with params from the model
    :param tau:
    """
    assert isinstance(tau, float)
    assert 0.0 < tau <= 1.0
    state = model.state_dict()
    tgt_state = target.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * tau + (1 - tau) * v
    target.load_state_dict(tgt_state)


class StratSyncVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(
        self, env_fns, num_rewards, observation_space=None, action_space=None, copy=True
    ):
        super().__init__(env_fns, observation_space, action_space, copy)
        self._rewards = np.zeros((self.num_envs, num_rewards), dtype=np.float64)


class StratAsyncVectorEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_fns,
        num_rewards,
        observation_space=None,
        action_space=None,
        metadata=None,
    ):
        super().__init__(
            env_fns,
            observation_space=observation_space,
            action_space=action_space,
            metadata=metadata,
        )
        self._rewards = np.zeros((self.num_envs, num_rewards), dtype=np.float64)


def make_env(
    arguments,
    idx,
    run_name,
    extra_wrapper=None,
):
    def thunk():
        env = gym.make(arguments.gym_id)
        if arguments.capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"monitor/{run_name}",
                    episode_trigger=lambda x: x % arguments.video_freq == 0,
                )
        # For PPO -------------------------------------------------------
        if "continuous" in arguments and arguments.continuous:
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )
        if "normalize" in arguments and arguments.normalize:
            env = gym.wrappers.NormalizeReward(env, gamma=arguments.gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        # ---------------------------------------------------------------
        if not ("hierarchical" in arguments and arguments.hierarchical):
            env = RecordEpisodeStatistics(env)
        if extra_wrapper is not None:
            env = extra_wrapper(env)
        env.seed(arguments.seed)
        env.action_space.seed(arguments.seed)
        env.observation_space.seed(arguments.seed)
        return env

    return thunk
