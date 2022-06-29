import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderStrat(LunarLander):
    def __init__(self, stratified=True):
        super().__init__()
        self.cumulative_reward_info = {
            "reward_Distance": 0,
            "reward_Speed": 0,
            "reward_Angle": 0,
            "reward_Left_contact": 0,
            "reward_Right_contact": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        self.stratified = stratified
        self.prev_rew = None

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
        shaping = np.zeros(8)
        strat_reward = np.zeros(8)
        # Distance to center
        shaping[0] = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        # Speed discount
        shaping[1] = -100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        # Angle discount
        shaping[2] = -100 * abs(state[4])
        # Ground Contacts
        shaping[5] = 10 * state[6]
        shaping[6] = 10 * state[7]

        # Win/Lost
        if done:
            self.prev_rew = None
            shaping = np.zeros(8)
            if self.game_over or abs(state[0]) >= 1.0:
                strat_reward[7] = -100
            if not self.lander.awake:
                strat_reward[7] = 100

        if self.prev_rew is not None:
            strat_reward = shaping - self.prev_rew
            # Power discount
            strat_reward[3] = -m_power * 0.30
            strat_reward[4] = -s_power * 0.03

        self.cumulative_reward_info["reward_Distance"] += strat_reward[0]
        self.cumulative_reward_info["reward_Speed"] += strat_reward[1]
        self.cumulative_reward_info["reward_Angle"] += strat_reward[2]
        self.cumulative_reward_info["reward_Power_linear"] += strat_reward[3]
        self.cumulative_reward_info["reward_Power_angular"] += strat_reward[4]
        self.cumulative_reward_info["reward_Left_contact"] += strat_reward[5]
        self.cumulative_reward_info["reward_Right_contact"] += strat_reward[6]
        self.cumulative_reward_info["reward_Goal"] += strat_reward[7]

        self.prev_rew = shaping
        self.prev_rew[3] = 0
        self.prev_rew[4] = 0

        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        # print("Original:", reward)
        # print("Stratified:", strat_reward.sum())
        reward = strat_reward if self.stratified else strat_reward.sum()
        if done:
            self.prev_rew = None
            self.cumulative_reward_info = {
                "reward_Distance": 0,
                "reward_Speed": 0,
                "reward_Angle": 0,
                "reward_Left_contact": 0,
                "reward_Right_contact": 0,
                "reward_Power_linear": 0,
                "reward_Power_angular": 0,
                "reward_Goal": 0,
                "Original_reward": 0,
            }
        return state, reward, done, info


class LunarLanderContinuousStrat(LunarLanderStrat):
    continuous = True
