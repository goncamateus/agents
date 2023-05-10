import gym
import envs


def main():
    env = gym.make("FrozenLake-v5")
    env.reset()
    for _ in range(1000):
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()


if __name__ == "__main__":
    main()
