import gym
from utils.wrappers import LunarLanderStratWrapper


def main():
    env = gym.make("LunarLanderContinuous-v2")
    env = LunarLanderStratWrapper(env)
    env.reset()
    for _ in range(1000):
        action = [0.1, 0.1]
        state, reward, done, info = env.step(action)
        env.render(mode='rgb_array')
        print("state")
        print(state)
        if done:
            env.reset()


if __name__ == "__main__":
    main()
