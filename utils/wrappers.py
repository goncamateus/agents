import numpy as np
import gym


class LunarLanderStratWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LunarLanderStratWrapper, self).__init__(env)
        self.cumulative_reward_info = {
            "Distance": 0,
            "Speed": 0,
            "Angle": 0,
            "Left_contact": 0,
            "Right_contact": 0,
            "Original_reward": 0,
        }

    def step(self, action):
        state, reward, done, info = super().step(action)
        strat_reward = np.zeros(5)
        # Distance to center
        strat_reward[0] = -np.sqrt(state[0] * state[0] + state[1] * state[1])
        self.cumulative_reward_info["Distance"] += strat_reward[0]
        # Speed discount
        strat_reward[1] = -np.sqrt(state[2] * state[2] + state[3] * state[3])
        self.cumulative_reward_info["Speed"] += strat_reward[1]
        # Angle discount
        strat_reward[2] = -abs(state[4])
        self.cumulative_reward_info["Angle"] += strat_reward[2]
        # Ground Contacts
        strat_reward[3] = state[6]
        self.cumulative_reward_info["Left_contact"] += strat_reward[3]
        strat_reward[4] = state[7]
        self.cumulative_reward_info["Right_contact"] += strat_reward[4]
        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        if done:
            self.cumulative_reward_info = {
                "Distance": 0,
                "Speed": 0,
                "Angle": 0,
                "Left_contact": 0,
                "Right_contact": 0,
                "Original_reward": 0,
            }
        return state, strat_reward, done, info
