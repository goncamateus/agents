import random

import numpy as np
import torch

from agents.utils.replay_buffer import ReplayBuffer


class ReplayWeightAwareBuffer(ReplayBuffer):
    def add(self, state, action, reward, next_state, done, weights):
        """
        Add an experience to the buffer.
        If the buffer is full, overwrite the oldest experience.
        This method accepts multiple experiences at once.
        
        Args:
            state (np.ndarray): The state(s) at the current time step.
            action (np.ndarray): The action(s) taken at the current time step.
            reward (np.ndarray): The reward(s) received at the current time step.
            next_state (np.ndarray): The state(s) at the next time step.
            done (np.ndarray): A boolean indicating if the episode is done.
            weights (np.ndarray): The weights of the experiences in the buffer.
            
        Example:
        ### Single experience
            >>> state = np.array([1, 2, 3])
            >>> action = np.array([0])
            >>> reward = np.array([1])
            >>> next_state = np.array([2, 3, 4])
            >>> done = np.array([False])
            >>> weights = np.array([[1,2]])
            >>> buffer.add(state, action, reward, next_state, done, weights)
        ### Multiple experiences
            >>> state = np.array([[1, 2, 3], [4, 5, 6]])
            >>> action = np.array([0, 1])
            >>> reward = np.array([1, 2])
            >>> next_state = np.array([[2, 3, 4], [5, 6, 7]])
            >>> done = np.array([False, True])
            >>> weights = np.array([[1, 2], [3, 4]])
            >>> buffer.add(state, action, reward, next_state, done, weights)
        """
        for i in range(len(state)):
            rew = reward[i]
            act = action[i]
            weight = weights[i]
            if rew.shape == ():
                rew = np.array([rew])
            if act.shape == ():
                act = np.array([act])
            if weight.shape == ():
                weight = np.array([weight])
            experience = (
                state[i],
                act,
                rew,
                next_state[i],
                done[i],
                weight,
            )
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
            self.ptr = int((self.ptr + 1) % self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer given a batch size.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            states_v (torch.Tensor): The states at the current time step.
            actions_v (torch.Tensor): The actions taken at the current time step.
            rewards_v (torch.Tensor): The rewards received at the current time step.
            last_states_v (torch.Tensor): The states at the next time step.
            dones_t (torch.Tensor): A boolean indicating if the episode is done.
            weights_v (torch.Tensor): The weights of the experiences in the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, weights = map(
            np.array, zip(*batch)
        )
        states_v = torch.Tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_v = torch.Tensor(rewards).to(self.device)
        last_states_v = torch.Tensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        weights_v = torch.Tensor(weights).to(self.device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t, weights_v
