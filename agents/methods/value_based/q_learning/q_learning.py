import pathlib
import numpy as np

from agents.methods.agent import Agent
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
from typing import TypeVar, Dict

T_space = TypeVar("T_space", Discrete, Box)


class QLearning(Agent):
    """Implementation of the Q-Learning algorithm using epsilon-greedy decay policy.

    Attributes:
        q_table (ArrayLike): The Q-table that stores the Q-values for each state-action pair.
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions in the environment.
        gamma (float): The discount factor.
        alpha (float): The learning rate.
        epsilon (float): The exploration rate.
        epsilon_decay (float): The decay rate of epsilon.
        epsilon_min (float): The minimum exploration rate.
        reward_scale (float): The scaling factor for the reward
    """

    def __init__(self, hyper_parameters, observation_space, action_space):
        super().__init__(hyper_parameters, observation_space, action_space)
        self.set_table()

    def hyper_parameters(self, hyper_parameters: Dict):
        """Set the hyperparameters of the agent.

        Args:
            hyper_parameters (dict): The hyperparameters for the agent.

        Raises:
            ValueError: When the hyperparameters are not set.
        """
        if not all(
            key in hyper_parameters
            for key in ["gamma", "alpha", "epsilon", "epsilon_decay", "epsilon_min"]
        ):
            raise ValueError("Hyperparameters not set.")
        self.gamma = hyper_parameters.get("gamma")
        self.alpha = hyper_parameters.get("alpha")
        self.epsilon = hyper_parameters.get("epsilon")
        self.epsilon_decay = hyper_parameters.get("epsilon_decay")
        self.epsilon_min = hyper_parameters.get("epsilon_min")
        self.reward_scale = hyper_parameters.get("reward_scale")

    def set_input_space(self, observation_space: T_space):
        """Set the input space of the agent.

        Args:
            observation_space (Discrete|Box): The observation space of the environment.

        Raises:
            ValueError: When the observation space is not of type Discrete or Box 2D.
        """
        self.observation_space = observation_space
        if isinstance(observation_space, Discrete):
            self.observation_space = observation_space
            self.num_states = observation_space.n
        elif isinstance(observation_space, Box):
            # Check if the Box observation space is 2D
            matrix_check = len(observation_space.shape) == 2
            if matrix_check:
                self.num_states = (
                    observation_space.shape[0] * observation_space.shape[1]
                )
            else:
                raise ValueError(
                    "When using a Box observation space, the shape must be 2D."
                )
        else:
            raise ValueError("Observation space must be of type Discrete or Box 2D.")

    def set_output_space(self, action_space: Discrete):
        """Set the output space of the agent.

        Args:
            action_space (Discrete): The action space of the environment.

        Raises:
            ValueError: When the action space is not of type Discrete.
        """
        if not isinstance(action_space, Discrete):
            raise ValueError("Action space must be of type Discrete.")
        self.action_space = action_space
        self.num_actions = action_space.n

    def set_table(self):
        """Initialize the Q-table with zeros."""
        ...

    def get_output(self, observation: int):
        """Get the output of the agent -> Argmax(Q(s, a)).

        Args:
            observation (int): The observation from the environment.

        Returns:
            int: The action to take.
        """
        ...

    def epsilon_greedy(self, observation: int):
        """Choose an action using epsilon-greedy policy.

        Args:
            observation (int): The observation from the environment.

        Returns:
            int: The action to take.
        """
        ...

    def get_action(self, observation: int):
        """Choose an action using epsilon-greedy decay policy.

        Args:
            observation (int): The observation from the environment.

        Returns:
            int: The action to take.
        """
        action = self.epsilon_greedy(observation)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Q-Learning algorithm.

        Args:
            state (int): observation from the environment.
            action (int): action taken by the agent.
            reward (float): reward received from the environment.
            next_state (int): next observation from the environment.

        Returns:
            float: The TD error value to log.
        """
        reward *= self.reward_scale
        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a'(Q(s', a')) - Q(s, a))
        td_value = reward + self.gamma * (
            self.q_table[next_state].max() - self.q_table[state][action]
        )
        # Q(s, a) = Q(s, a) + alpha * TD
        self.q_table[state][action] += self.alpha * td_value
        return td_value

    def save(self, path: str):
        """Save the Q-table to a file.

        Args:
            path (str): The path to save the Q-table.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        if not isinstance(self.q_table, np.ndarray):
            self.q_table = np.array(self.q_table)
        np.save(path + "q_table.npy", self.q_table)

    def load(self, path: str):
        """Load the Q-table from a file.

        Args:
            path (str): The path to load the Q-table.
        """
        self.q_table = np.load(path + "q_table.npy")
