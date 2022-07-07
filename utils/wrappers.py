import gym
import numpy as np
import time
from collections import deque
from gym.wrappers.normalize import RunningMeanStd


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.num_rewards = getattr(env, "num_rewards", 1)
        self.weights = getattr(env, "ori_weights", None)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_returns_strat = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.return_strat_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_returns_strat = np.zeros(
            (self.num_envs, self.num_rewards), dtype=np.float32
        )
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super(RecordEpisodeStatistics, self).step(
            action
        )
        # strat = not isinstance(rewards, float) or isinstance(rewards, int)
        # if not strat:
        #     self.episode_returns += rewards
        # else:
        if not self.is_vector_env:
            rewards = rewards.reshape((1, -1))
        self.episode_returns += (rewards * self.weights).sum()
        self.episode_returns_strat += rewards * self.weights
        # Changes based on the experiment
        rewards = (rewards * self.weights).sum()
        self.episode_lengths += 1
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                # if strat:
                for j in range(self.num_rewards):
                    episode_info["component_%d" % j] = self.episode_returns_strat[i, j]
                self.return_strat_queue.append(episode_info["component_%d" % j])
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.episode_returns_strat[i] = np.zeros(
                    self.num_rewards, dtype=np.float32
                )
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )


class RewardTransformer:
    def __init__(
        self,
        args,
        epsilon=1e-8,
    ):
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros((args.num_envs, args.num_rewards))
        self.gamma = args.gamma
        self.epsilon = epsilon

    def transform(self, rews, dones):
        self.returns = self.returns * self.gamma + rews
        self.return_rms.update(self.returns)
        rews = rews / np.sqrt(self.return_rms.var + self.epsilon)
        self.returns[dones] = 0.0
        return np.clip(rews, -10, 10)
