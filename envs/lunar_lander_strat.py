import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderstrat(LunarLander):
    def __init__(self):
        super().__init__()
        self.cumulative_reward_info = {
            "Distance_x": 0,
            "Distance_y": 0,
            "Speed_x": 0,
            "Speed_y": 0,
            "Angle": 0,
            "Left_contact": 0,
            "Right_contact": 0,
            "Power_linear": 0,
            "Power_angular": 0,
            "Goal": 0,
            "Original_reward": 0,
        }
        self.prev_rew = np.zeros(10)

    def step(self, action):
        m_power = int(action == 2)
        s_power = int(action in [1, 3])
        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(10)
        # Distance to center
        strat_reward[0] = -abs(state[0]) - self.prev_rew[0]
        self.prev_rew[0] = -abs(state[0])
        strat_reward[1] = -abs(state[1]) - self.prev_rew[1]
        self.prev_rew[1] = -abs(state[1])
        self.cumulative_reward_info["Distance_x"] += strat_reward[0]
        self.cumulative_reward_info["Distance_y"] += strat_reward[1]
        # Speed discount
        strat_reward[2] = -abs(state[2]) - self.prev_rew[2]
        self.prev_rew[2] = -abs(state[2])
        strat_reward[3] = -abs(state[3]) - self.prev_rew[3]
        self.prev_rew[3] = -abs(state[3])
        self.cumulative_reward_info["Speed_x"] += strat_reward[2]
        self.cumulative_reward_info["Speed_y"] += strat_reward[3]
        # Angle discount
        strat_reward[4] = -abs(state[4]) - self.prev_rew[4]
        self.prev_rew[4] = -abs(state[4])
        self.cumulative_reward_info["Angle"] += strat_reward[4]
        # Power discount
        strat_reward[5] = -m_power - self.prev_rew[5]
        self.prev_rew[5] = -m_power
        strat_reward[6] = -s_power - self.prev_rew[6]
        self.prev_rew[6] = -s_power
        self.cumulative_reward_info["Power_linear"] += strat_reward[5]
        self.cumulative_reward_info["Power_angular"] += strat_reward[6]
        # Ground Contacts
        strat_reward[7] = state[6] - self.prev_rew[7]
        self.prev_rew[7] = state[6]
        strat_reward[8] = state[7] - self.prev_rew[8]
        self.prev_rew[8] = state[7]
        self.cumulative_reward_info["Left_contact"] += strat_reward[7]
        self.cumulative_reward_info["Right_contact"] += strat_reward[8]

        # Win/Lost
        if done:
            if not self.lander.awake:
                strat_reward[9] = 1
            else:
                strat_reward[9] = -1
        self.cumulative_reward_info["Goal"] += strat_reward[9]

        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        if done:
            self.prev_rew = np.zeros(10)
            self.cumulative_reward_info = {
                "Distance_x": 0,
                "Distance_y": 0,
                "Speed_x": 0,
                "Speed_y": 0,
                "Angle": 0,
                "Left_contact": 0,
                "Right_contact": 0,
                "Power_linear": 0,
                "Power_angular": 0,
                "Goal": 0,
                "Original_reward": 0,
            }
        return state, reward, done, info
