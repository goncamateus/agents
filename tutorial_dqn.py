import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_size, observation_space, action_space, device):
        self.max_size = max_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.mem_count = 0
        if isinstance(self.observation_space, gym.spaces.Discrete):
            self.state_buffer = torch.zeros(
                (max_size, observation_space.n), dtype=torch.float32
            )
            self.next_state_buffer = torch.zeros(
                (max_size, observation_space.n), dtype=torch.float32
            )
        else:
            self.state_buffer = torch.zeros(
                (max_size, observation_space.shape[0]),
                dtype=torch.float32,
            )
            self.next_state_buffer = torch.zeros(
                (max_size, observation_space.shape[0]),
                dtype=torch.float32,
            )
        self.action_buffer = torch.zeros((max_size, 1), dtype=torch.long)
        self.reward_buffer = torch.zeros((max_size, 1), dtype=torch.float32)
        self.done_buffer = torch.zeros(max_size, dtype=torch.bool)

    def to(self, device):
        self.device = device
        self.state_buffer = self.state_buffer.to(self.device)
        self.next_state_buffer = self.next_state_buffer.to(self.device)
        self.action_buffer = self.action_buffer.to(self.device)
        self.reward_buffer = self.reward_buffer.to(self.device)
        self.done_buffer = self.done_buffer.to(self.device)

    def add(self, observation, action, reward, next_observation, done):
        observation = torch.from_numpy(observation).float().to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_observation = torch.from_numpy(next_observation).float().to(self.device)

        self.state_buffer[self.mem_count % self.max_size] = observation
        self.action_buffer[self.mem_count % self.max_size] = action
        self.reward_buffer[self.mem_count % self.max_size] = reward
        self.next_state_buffer[self.mem_count % self.max_size] = next_observation
        self.done_buffer[self.mem_count % self.max_size] = done
        self.mem_count += 1

    def sample(self, batch_size=256):
        max_mem = min(self.mem_count, self.max_size)
        batch = torch.randint(0, max_mem, (batch_size,), device=self.device)
        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        next_states = self.next_state_buffer[batch]
        dones = self.done_buffer[batch]
        return states, actions, rewards, next_states, dones

    def size(self):
        return min(self.mem_count, self.max_size)


class QLearningAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        alpha=1e-3,
        gamma=0.99,
        buffer_max_len=10000,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.n_updates = 0
        self.device = torch.device("cpu")
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_max_len,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )
        self.set_net()

    def set_net(self):
        obs_size = (
            self.observation_space.n
            if isinstance(self.observation_space, gym.spaces.Discrete)
            else self.observation_space.shape[0]
        )
        self.dqn = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_space.n),
        )
        self.target_dqn = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_space.n),
        )
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.alpha)

    def to(self, device):
        self.device = device
        self.dqn.to(self.device)
        self.target_dqn.to(self.device)
        self.replay_buffer.to(self.device)

    def get_action(self, observation):
        with torch.no_grad():
            observation = torch.from_numpy(observation).float().to(self.device)
            q_values = self.dqn(observation)
            action = torch.argmax(q_values).item()
        return action

    def observe(self, observation, action, reward, next_observation, done):
        self.replay_buffer.add(observation, action, reward, next_observation, done)

    def update_policy(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1).values.unsqueeze(1)
            next_q_values[dones] = 0.0
        y = rewards + self.gamma * next_q_values
        q_values = self.dqn(states)
        q_values = q_values.gather(1, actions)
        loss = F.mse_loss(q_values, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1
        if self.n_updates % 10:
            self.target_dqn.load_state_dict(self.dqn.state_dict())


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    agent = QLearningAgent(env.observation_space, env.action_space)

    for episodes in range(1000):
        obs, info = env.reset()
        observation = np.zeros(agent.observation_space.n, dtype=np.float32)
        observation[obs] = 1.0
        done = False
        while not done:
            if np.random.random() < 0.85:
                action = agent.get_action(observation)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            next_observation = np.zeros(agent.observation_space.n, dtype=np.float32)
            next_observation[next_obs] = 1.0
            agent.observe(observation, action, reward, next_observation, done)
            observation = next_observation
        if agent.replay_buffer.size() >= 256:
            agent.update_policy(batch_size=256)

    env.close()


if __name__ == "__main__":
    main()
