import gym
import numpy as np

import envs


def main():
    env = gym.make("HierarchicalFrozenLake11x11Strat-v0")
    env.reset()
    last_manager_action = env.action_space.sample()["manager"]
    for _ in range(1000):
        manager_action = last_manager_action
        manager_x, manager_y = manager_action // np.sqrt(
            env.action_space["manager"].n
        ), manager_action % np.sqrt(env.action_space["manager"].n)
        manager_x = np.clip(
            manager_x + np.random.randint(-2, 2), 0, env.action_space["manager"].n - 1
        )
        manager_y = np.clip(
            manager_y + np.random.randint(-2, 2), 0, env.action_space["manager"].n - 1
        )
        rows = np.sqrt(env.action_space["manager"].n)
        manager_action = manager_x * rows + manager_y
        if manager_action < 0:
            manager_action = last_manager_action
        manager_action = np.clip(manager_action, 0, env.action_space["manager"].n)
        manager_action = int(manager_action)
        last_manager_action = manager_action
        action = {
            "manager": manager_action,
            "worker": env.action_space.sample()["worker"],
        }
        state, reward, done, info = env.step(action)
        print("State")
        print(state)

        print("Reward")
        print(reward)

        print("Info")
        print(info)

        env.render()
        import ipdb

        ipdb.set_trace()
        if done:
            last_manager_action = None
            env.reset()


if __name__ == "__main__":
    main()
