import gymnasium as gym
import numpy as np

import envs


class DyLamQLearning:
    def __init__(
        self, observation_space, action_space, n_rewards, alpha=1e-1, gamma=0.9999
    ):
        self.obs_size = observation_space.n
        self.action_size = action_space.n
        self.n_rewards = n_rewards
        self.q_table = np.zeros((self.obs_size, self.action_size))
        self.components_q = np.zeros(
            (
                self.n_rewards,
                self.obs_size,
                self.action_size,
            )
        )
        self.lambdas = np.ones(self.n_rewards) / self.n_rewards
        self.rb_reward = np.zeros((10, self.n_rewards))
        self.rb_idx = 0
        self.can_dylam = False
        self.rew_tau = 0.995
        self.rew_max = np.array([0, 1, 0])
        self.rew_min = np.array([-1, 0, -1])
        self.last_rew_mean = None
        self.alpha = alpha
        self.gamma = gamma

    def store_reward(self, reward):
        self.rb_reward[self.rb_idx] = reward
        self.rb_idx = (self.rb_idx + 1) % self.rb_reward.shape[0]
        self.can_dylam = self.rb_idx == 0 or self.can_dylam

    def dylam(self):
        if self.can_dylam:
            rew_mean_t = np.mean(self.rb_reward, axis=0)
            if self.last_rew_mean is not None:
                rew_mean_t = (
                    rew_mean_t + (self.last_rew_mean - rew_mean_t) * self.rew_tau
                )
            dQ = np.clip(
                (self.rew_max - rew_mean_t) / (self.rew_max - self.rew_min), 0, 1
            )
            expdQ = np.exp(dQ) - 1
            self.lambdas = expdQ / (np.sum(expdQ, 0) + 1e-4)
            self.last_rew_mean = rew_mean_t

    def get_action(self, observation):
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def update_component_tables(self, observation, action, reward, next_obs):
        for i in range(self.n_rewards):
            update = reward[i] + self.gamma * (
                self.components_q[i][next_obs].max()
                - self.components_q[i][observation][action]
            )
            self.components_q[i][observation][action] = (
                self.components_q[i][observation][action] + self.alpha * update
            )

    def update_policy(self, observation, action, reward, next_obs):
        self.update_component_tables(observation, action, reward, next_obs)
        Qs = self.components_q[:, observation, action]
        self.q_table[observation][action] = (Qs * self.lambdas).sum()


def main():
    env = gym.make("TaxiStrat-v0", render_mode="ansii")
    agent = DyLamQLearning(env.observation_space, env.action_space, n_rewards=3)
    obs, info = env.reset()
    reward = 0
    for episodes in range(1000):
        obs, info = env.reset()
        done = False
        truncated = False
        epi_reward = 0
        while not (done or truncated):
            if np.random.random() < 0.85:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            epi_reward += (reward * np.array([200, 20, 10])).sum()
            if done:
                next_obs = obs
            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs
        print(f"Episode {episodes} reward: {epi_reward}")
        agent.store_reward(epi_reward)
        agent.dylam()

    env.close()

    print("---------------Evaluating---------------")
    env = gym.make("TaxiStrat-v0", render_mode="ansii")
    for episodes in range(10):
        obs, info = env.reset()
        done = False
        truncated = False
        epi_reward = 0
        while not (done or truncated):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            epi_reward += (reward * np.array([200, 20, 10])).sum()
        print(f"Evaluation Episode {episodes} reward: {epi_reward}")
    env.close()


if __name__ == "__main__":
    main()
