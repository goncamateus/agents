import gym


class HalfCheetahStrat(gym.Wrapper):
    def __init__(self, env):
        super(HalfCheetahStrat, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
