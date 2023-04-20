import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderStratV2(LunarLander):
    def __init__(self, stratified=True):
        super().__init__()
        self.cumulative_reward_info = {
            "reward_Shaped": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        self.stratified = stratified
        self.prev_rew = np.zeros(10)
        self.num_rewards = 4
        self.ori_weights = np.array([100.0, 0.3, 0.03, 100.0])

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                s_power = 1.0

        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(10)
        # Distance to center
        strat_reward[0] = -abs(state[0]) - self.prev_rew[0]
        self.prev_rew[0] = -abs(state[0])
        strat_reward[1] = -abs(state[1]) - self.prev_rew[1]
        self.prev_rew[1] = -abs(state[1])
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[0]
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[1]
        # Speed discount
        strat_reward[2] = -abs(state[2]) - self.prev_rew[2]
        self.prev_rew[2] = -abs(state[2])
        strat_reward[3] = -abs(state[3]) - self.prev_rew[3]
        self.prev_rew[3] = -abs(state[3])
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[2]
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[3]
        # Angle discount
        strat_reward[4] = -abs(state[4]) - self.prev_rew[4]
        self.prev_rew[4] = -abs(state[4])
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[4]
        # Power discount
        strat_reward[5] = -m_power - self.prev_rew[5]
        self.prev_rew[5] = -m_power
        strat_reward[6] = -s_power - self.prev_rew[6]
        self.prev_rew[6] = -s_power
        self.cumulative_reward_info["reward_Power_linear"] += strat_reward[5]
        self.cumulative_reward_info["reward_Power_angular"] += strat_reward[6]
        # Ground Contacts
        strat_reward[7] = state[6] - self.prev_rew[7]
        self.prev_rew[7] = state[6]
        strat_reward[8] = state[7] - self.prev_rew[8]
        self.prev_rew[8] = state[7]
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[7]
        self.cumulative_reward_info["reward_Shaped"] += strat_reward[8]

        # Win/Lost
        if done:
            if not self.lander.awake:
                strat_reward[9] = 1
            else:
                strat_reward[9] = -1

        self.cumulative_reward_info["reward_Goal"] += strat_reward[9]
        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)

        if reward == 0:
            strat_reward = np.zeros(10)

        mo_strat_reward = np.zeros(self.num_rewards)
        mo_strat_reward[0] = strat_reward[:5].sum() + strat_reward[7:9].sum()
        mo_strat_reward[1] = strat_reward[5]
        mo_strat_reward[2] = strat_reward[6]
        mo_strat_reward[3] = strat_reward[9]

        if not self.stratified:
            mo_strat_reward = mo_strat_reward.sum()

        if done:
            self.prev_rew = np.zeros(10)
            self.cumulative_reward_info = {
                "reward_Shaped": 0,
                "reward_Power_linear": 0,
                "reward_Power_angular": 0,
                "reward_Goal": 0,
                "Original_reward": 0,
            }
        return state, mo_strat_reward, done, info


class LunarLanderContinuousStratV2(LunarLanderStratV2):
    continuous = True
