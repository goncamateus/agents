import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(self, observation_space, action_space, alpha=1e-1, gamma=0.99):
        self.obs_size = observation_space.n
        self.action_size = action_space.n
        self.q_table = np.zeros((self.obs_size, self.action_size))
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, observation):
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def update_policy(self, observation, action, reward, next_obs):
        update = reward + self.gamma * (
            self.q_table[next_obs].max() - self.q_table[observation][action]
        )
        self.q_table[observation][action] = (
            self.q_table[observation][action] + self.alpha * update
        )


def main():
    env = gym.make("Taxi-v3", render_mode="ansii")
    agent = QLearningAgent(env.observation_space, env.action_space)
    obs, info = env.reset()
    reward = 0

    for episodes in range(1000):
        obs, info = env.reset()
        done = False
        epi_reward = 0
        while not done:
            if np.random.random() < 0.85:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            epi_reward += reward
            if done:
                next_obs = obs
            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs
        print(f"Episode {episodes} reward: {epi_reward}")

    env.close()

    print("---------------Evaluating---------------")
    env = gym.make("Taxi-v3", render_mode="ansii")
    for episodes in range(10):
        obs, info = env.reset()
        done = False
        truncated = False
        epi_reward = 0
        while not (done or truncated):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            epi_reward += reward
        print(f"Evaluation Episode {episodes} reward: {epi_reward}")
    env.close()


if __name__ == "__main__":
    main()
