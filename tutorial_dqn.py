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
            self.state_buffer = np.zeros(
                (max_size, observation_space.n), dtype=np.float32
            )
            self.next_state_buffer = np.zeros(
                (max_size, observation_space.n), dtype=np.float32
            )
        else:
            self.state_buffer = np.zeros(
                (max_size, observation_space.shape[0]),
                dtype=np.float32,
            )
            self.next_state_buffer = np.zeros(
                (max_size, observation_space.shape[0]),
                dtype=np.float32,
            )
        self.action_buffer = np.zeros((max_size, 1), dtype=np.int64)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros(max_size, dtype=bool)

    def to(self, device):
        self.device = device

    def add(self, observation, action, reward, next_observation, done):
        reward =np.array([reward])
        self.state_buffer[self.mem_count % self.max_size] = observation
        self.action_buffer[self.mem_count % self.max_size] = action
        self.reward_buffer[self.mem_count % self.max_size] = reward
        self.next_state_buffer[self.mem_count % self.max_size] = next_observation
        self.done_buffer[self.mem_count % self.max_size] = done
        self.mem_count += 1

    def sample(self, batch_size=256):
        max_mem = min(self.mem_count, self.max_size)
        batch = np.random.randint(0, max_mem, (batch_size,))
        states = torch.from_numpy(self.state_buffer[batch]).to(self.device)
        actions = torch.from_numpy(self.action_buffer[batch]).to(self.device)
        rewards = torch.from_numpy(self.reward_buffer[batch]).to(self.device)
        next_states = torch.from_numpy(self.next_state_buffer[batch]).to(self.device)
        dones = torch.from_numpy(self.done_buffer[batch]).to(self.device)
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
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
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

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            q_values = self.dqn(torch.FloatTensor(state).to(self.device))
            action = q_values.argmax().item()
        else:
            action = self.action_space.sample()
            self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)
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
    env = gym.make("MountainCar-v0", render_mode="human")
    agent = QLearningAgent(env.observation_space, env.action_space)
    agent.to("cuda" if torch.cuda.is_available() else "cpu")
    for episode in range(1000):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, (done or truncated))
            obs = next_obs
        print(f"Episode {episode}:", reward)
        if agent.replay_buffer.size() >= 256:
            agent.update_policy(batch_size=256)

    env.close()


if __name__ == "__main__":
    main()
