import gym
import numpy as np

import envs


def main():
    env = gym.make("FrozenLake11x11Strat-v0")
    env.reset()
    for step in range(100):
        # env.render()
        # action = input("UP = 0; DOWN = 1; RIGHT = 2; LEFT = 3\n")
        # action = int(action)
        # if step < 5:
        #     action = 0
        # else:
        #     action = 2
        state, reward, done, info = env.step(3)
        # print("Reward")
        # print(reward)


        if done:
            print("Info")
            print(info)
            print("Done")
            env.reset()


if __name__ == "__main__":
    main()
