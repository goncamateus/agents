from abc import abstractmethod

import numpy as np
from gymnasium.spaces.space import Space

from agents.common.agent import Agent


class SAC(Agent):
    """SAC agent.

    Args:
        Agent (_type_): _description_
    """

    def __init__(self, hyper_parameters, observation_space, action_space):
        super().__init__(hyper_parameters, observation_space, action_space)
        self.build_networks()
        self.set_target_networks()
        self.build_optimizers()
        self.set_device()
        self.init_replay_buffer()
        self.checkup()

    def hyper_parameters(
        self,
        gamma: float,
        device: str,
        reward_scale: float,
        buffer_size: int,
        hidden_dim: int,
        q_learning_rate: float,
        policy_learning_rate: float,
        log_sig_min: float = -5,
        log_sig_max: float = 2,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        epsilon: float = 1e-6,
        action_range: tuple = None,
    ) -> None:
        """Set the hyperparameters of the agent.

        Args:
            gamma (float): The discount factor.
            device (str): The device to use.
            reward_scale (float): The reward scaling factor.
            buffer_size (int): The size of the replay buffer
            hidden_dim (int): The number of units in the hidden layers.
            q_learning_rate (float): The learning rate for the Q-network.
            policy_learning_rate (float): The learning rate for the policy network.
            log_sig_min (float, optional): The minimum value for the log standard deviation. Defaults to -5.
            log_sig_max (float, optional): The maximum value for the log standard deviation. Defaults to 2.
            alpha (float, optional): The entropy coefficient. Defaults to 0.2.
            automatic_entropy_tuning (bool): Whether to automatically tune the entropy coefficient.
            epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
            action_range (tuple, optional): The range of the action space. Defaults to None.

        Raises:
            ValueError: When the hyperparameters are not set.
        """
        self.gamma = gamma
        self.device = device
        self.reward_scale = reward_scale
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.action_range = action_range

    def set_input_space(self, observation_space: Space):
        """Set the input space of the agent.

        Args:
            observation_space (Space): The observation space.
        """
        self.num_inputs = np.array(observation_space.shape).prod()

    def set_output_space(self, action_space: Space):
        """Set the output space of the agent.

        Args:
            action_space (Space): The action space.
        """
        self.num_outpts = np.array(action_space.shape).prod()
        self.action_range = (action_space.high, action_space.low)

    def checkup(self):
        """Check SAC networks and optimizers are set."""
        assert self.actor is not None, "Actor network is not set."
        assert self.critic is not None, "Critic networks are not set."
        assert self.actor_optimizer is not None, "Actor optimizer is not set."
        assert self.critic_optimizer is not None, "Critic optimizers are not set."
        if self.automatic_entropy_tuning:
            assert self.target_entropy is not None, "Target entropy is not set."
            assert self.alpha_optimizer is not None, "Alpha optimizer is not set."
        assert self.target_actor is not None, "Target actor network is not set."
        assert self.target_critic is not None, "Target critic networks are not set."

    @abstractmethod
    def set_device(self): ...

    @abstractmethod
    def build_networks(self): ...

    @abstractmethod
    def set_target_networks(self): ...

    @abstractmethod
    def build_optimizers(self): ...

    @abstractmethod
    def init_replay_buffer(self): ...

    def get_action(self, observations: np.ndarray, deterministic: bool = False):
        """Get the action from the agent.

        Args:
            observations (np.ndarray): The observations.
            deterministic (bool, optional): Whether to use a deterministic policy. Defaults to False.

        Returns:
            np.ndarray: The action.
        """
        ...

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        """Update the critic network.

        Args:
            state_batch (Any): The state batch.
            action_batch (Any): The action batch.
            reward_batch (Any): The reward batch.
            next_state_batch (Any): The next state batch.
            done_batch (Any): The done batch.
        """
        ...

    def update_actor(self, state_batch):
        """Update the actor network.

        Args:
            state_batch (Any): The state batch.
        """
        ...

    def update_alpha(self, state_batch):
        """Update the entropy coefficient.

        Args:
            state_batch (Any): The state batch.
        """
        ...

    def update(self, batch_size: int, update_actor: bool = True):
        """Update the agent.

        Args:
            batch_size (int): The batch size.
            update_actor (bool, optional): Whether to update the actor. Defaults to True.
        """
        ...

    def save(self, path: str): ...

    def load(self, path: str): ...
