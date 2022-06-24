import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderStrat(LunarLander):
    def __init__(self, stratified=True):
        super().__init__()
        self.cumulative_reward_info = {
            "Distance": 0,
            "Speed": 0,
            "Angle": 0,
            "Left_contact": 0,
            "Right_contact": 0,
            "Power_linear": 0,
            "Power_angular": 0,
            "Goal": 0,
            "Original_reward": 0,
        }
        self.stratified = stratified
        self.prev_rew = np.zeros(8)

    def step(self, action):
        m_power = int(action == 2)
        s_power = int(action in [1, 3])
        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(8)
        # Distance to center
        strat_reward[0] = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        strat_reward[0] = strat_reward[0] - self.prev_rew[0]
        self.prev_rew[0] = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        self.cumulative_reward_info["Distance"] += strat_reward[0]
        # Speed discount
        strat_reward[1] = -100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        strat_reward[1] = strat_reward[1] - self.prev_rew[1]
        self.prev_rew[1] = -100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        self.cumulative_reward_info["Speed"] += strat_reward[1]
        # Angle discount
        strat_reward[2] = -100 * abs(state[4])
        strat_reward[2] = strat_reward[2] - self.prev_rew[2]
        self.prev_rew[2] = -100 * abs(state[4])
        self.cumulative_reward_info["Angle"] += strat_reward[2]
        # Power discount
        strat_reward[3] = -m_power - self.prev_rew[3]
        self.prev_rew[3] = -m_power
        strat_reward[4] = -s_power - self.prev_rew[6]
        self.prev_rew[4] = -s_power
        self.cumulative_reward_info["Power_linear"] += strat_reward[3]
        self.cumulative_reward_info["Power_angular"] += strat_reward[4]
        # Ground Contacts
        strat_reward[5] = 10 * state[6] - self.prev_rew[5]
        self.prev_rew[5] = 10 * state[6]
        strat_reward[6] = 10 * state[7] - self.prev_rew[6]
        self.prev_rew[6] = 10 * state[7]
        self.cumulative_reward_info["Left_contact"] += strat_reward[5]
        self.cumulative_reward_info["Right_contact"] += strat_reward[6]

        # Win/Lost
        if done:
            if not self.lander.awake:
                strat_reward[7] = 100
            else:
                strat_reward[7] = -100
        self.cumulative_reward_info["Goal"] += strat_reward[7]

        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        if done:
            self.prev_rew = np.zeros(8)
            self.cumulative_reward_info = {
                "Distance": 0,
                "Speed": 0,
                "Angle": 0,
                "Left_contact": 0,
                "Right_contact": 0,
                "Power_linear": 0,
                "Power_angular": 0,
                "Goal": 0,
                "Original_reward": 0,
            }
        reward = strat_reward if self.stratified else reward
        return state, reward, done, info


class LunarLanderContinuousStrat(LunarLanderStrat):
    continuous = True
