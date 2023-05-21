import gym
import numpy as np

import envs


def main():
    env = gym.make("HierarchicalSimpleNav-v0")
    env.reset()
    for step in range(10000):
        # env.render()
        # action = input("UP = 0; DOWN = 1; RIGHT = 2; LEFT = 3\n")
        # action = int(action)
        # if step < 5:
        #     action = 0
        # else:
        #     action = 2
        state, reward, done, info = env.step(env.action_space.sample())
        print("Reward")
        print(reward)

        print("State")
        print(state)

        if done:
            print("Info")
            print(info)
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
