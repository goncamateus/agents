import numpy as np
from abc import ABC


class BaseReplayBuffer(ABC):
    """
    BaseReplayBuffer is a class that implements a replay buffer for storing and sampling experiences in reinforcement learning.

    Attributes:
        max_size (int): The maximum size of the buffer.
        device (str): The device to store the tensors on ("cpu" or "cuda").
        buffer (list): The list to store experiences.
        ptr (int): The pointer to the current position in the buffer.
    """

    def __init__(self, max_size, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
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

        Example:
        ### Single experience
            >>> state = np.array([1, 2, 3])
            >>> action = np.array([0])
            >>> reward = np.array([1])
            >>> next_state = np.array([2, 3, 4])
            >>> done = np.array([False])
            >>> buffer.add(state, action, reward, next_state, done)
        ### Multiple experiences
            >>> state = np.array([[1, 2, 3], [4, 5, 6]])
            >>> action = np.array([0, 1])
            >>> reward = np.array([1, 2])
            >>> next_state = np.array([[2, 3, 4], [5, 6, 7]])
            >>> done = np.array([False, True])
            >>> buffer.add(state, action, reward, next_state, done)
        """
        for i in range(len(state)):
            rew = reward[i]
            act = action[i]
            if rew.shape == ():
                rew = np.array([rew])
            if act.shape == ():
                act = np.array([act])
            experience = (
                state[i],
                act,
                rew,
                next_state[i],
                done[i],
            )
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
            self.ptr = int((self.ptr + 1) % self.max_size)

    def clear(self):
        """Clear the buffer and reset the pointer"""
        self.buffer.clear()
        self.ptr = 0

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

        Example:
            >>> states_v, actions_v, rewards_v, last_states_v, dones_t = buffer.sample(32)
        """
        ...

    def __len__(self):
        return len(self.buffer)
