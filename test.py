import gym
import envs


def main():
    env = gym.make("FrozenLake13x13Strat-v0")
    env.reset()
    for _ in range(1000):
        state, reward, done, info = env.step(env.action_space.sample())
        print("State")
        print(state)

        print("Reward")
        print(reward)

        print("Info")
        print(info)

        env.render()
        if done:
            env.reset()


if __name__ == "__main__":
    main()
