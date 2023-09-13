# Credits: 
# Farama Foundation
# https://github.com/Farama-Foundation/MO-Gymnasium/blob/main/mo_gymnasium/envs/highway/highway.py
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from highway_env.envs import HighwayEnv, HighwayEnvFast


class ModHighwayEnv(HighwayEnv, EzPickle):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.cumulative_reward_info = {
            "reward_high_speed": 0,
            "reward_right_lane": 0,
            "reward_collision": 0,
            "Original_reward": 0,
        }
    
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        state, info = super().reset(seed=seed)
        self.cumulative_reward_info = {
            "reward_high_speed": 0,
            "reward_right_lane": 0,
            "reward_collision": 0,
            "Original_reward": 0,
        }
        info.update(self.cumulative_reward_info)
        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        self.cumulative_reward_info["reward_high_speed"] += rewards["high_speed_reward"]
        self.cumulative_reward_info["reward_right_lane"] += rewards["right_lane_reward"]
        self.cumulative_reward_info["reward_collision"] += -rewards["collision_reward"]
        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        return obs, reward, terminated, truncated, info
    
class ModHighwayEnvFast(HighwayEnvFast, EzPickle):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.cumulative_reward_info = {
            "reward_high_speed": 0,
            "reward_right_lane": 0,
            "reward_collision": 0,
            "Original_reward": 0,
        }
    
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        state, info = super().reset(seed=seed)
        self.cumulative_reward_info = {
            "reward_high_speed": 0,
            "reward_right_lane": 0,
            "reward_collision": 0,
            "Original_reward": 0,
        }
        info.update(self.cumulative_reward_info)
        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        self.cumulative_reward_info["reward_high_speed"] += rewards["high_speed_reward"]
        self.cumulative_reward_info["reward_right_lane"] += rewards["right_lane_reward"]
        self.cumulative_reward_info["reward_collision"] += -rewards["collision_reward"]
        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        return obs, reward, terminated, truncated, info

class HighwayStratEnv(ModHighwayEnv):
    """
    ## Description
    Multi-objective version of the HighwayEnv environment.

    See [highway-env](https://github.com/eleurent/highway-env) for more information.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: high speed reward
    - 1: right lane reward
    - 2: collision reward
    """

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"]/40,
                rewards["right_lane_reward"]/40,
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info


class HighwayStratEnvFast(ModHighwayEnvFast):
    """A multi-objective version of the HighwayFastEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"]/40,
                rewards["right_lane_reward"]/40,
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info