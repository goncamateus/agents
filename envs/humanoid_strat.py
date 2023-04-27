import numpy as np
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv


class HumanoidStratEnv(HumanoidEnv):
    def __init__(
        self,
        xml_file="humanoid.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        stratified=True,
    ):
        self.stratified = stratified
        self.ori_weights = np.array(
            [
                forward_reward_weight,
                healthy_reward,
                ctrl_cost_weight,
                # contact_cost_weight,
            ]
        )
        self.num_rewards = 3
        self.cumulative_reward_info = {
            "reward_linvel": 0,
            "reward_quadctrl": 0,
            "reward_alive": 0,
            "Original_reward": 0,
        }
        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=contact_cost_weight,
            contact_cost_range=contact_cost_range,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )

    def reset(self):
        self.cumulative_reward_info = {
            "reward_linvel": 0,
            "reward_quadctrl": 0,
            "reward_alive": 0,
            "Original_reward": 0,
        }
        return super().reset()

    def step(self, action):
        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(3)
        # Forward reward
        strat_reward[0] = info["reward_linvel"]
        # Alive reward
        strat_reward[1] = info["reward_alive"]
        # Quadctrl reward
        strat_reward[2] = info["reward_quadctrl"]

        strat_reward = strat_reward / self.ori_weights

        self.cumulative_reward_info["reward_linvel"] += strat_reward[0]
        self.cumulative_reward_info["reward_alive"] += strat_reward[1]
        self.cumulative_reward_info["reward_quadctrl"] += strat_reward[2]
        self.cumulative_reward_info["Original_reward"] += reward
        # Scaling the reward to [-1, 1] means random agent in this environment

        info.update(self.cumulative_reward_info)
        return state, strat_reward, done, info
