import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HalfCheetahStratEnv(HalfCheetahEnv):
    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        stratified=True,
        **kwargs
    ):
        self.stratified = stratified
        self.ori_weights = np.array([forward_reward_weight, ctrl_cost_weight])
        self.scale = np.array([1, 1])
        self.num_rewards = 2
        self.cumulative_reward_info = {
            "reward_run": 0,
            "reward_ctrl": 0,
            "Original_reward": 0,
        }
        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            **kwargs
        )

    def step(self, action):
        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(2)
        # Forward reward
        strat_reward[0] = info["reward_run"]
        # Control reward
        strat_reward[1] = info["reward_ctrl"]
        strat_reward = strat_reward / self.ori_weights

        self.cumulative_reward_info["reward_run"] += strat_reward[0]
        self.cumulative_reward_info["reward_ctrl"] += strat_reward[1]
        self.cumulative_reward_info["Original_reward"] += reward
        reward = strat_reward

        info.update(self.cumulative_reward_info)

        if done:
            self.cumulative_reward_info = {
                "reward_run": 0,
                "reward_ctrl": 0,
                "Original_reward": 0,
            }
        return state, reward, done, info
