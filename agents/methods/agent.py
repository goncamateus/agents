from abc import ABC, abstractmethod
from typing import Dict

from gymnasium.spaces.space import Space


class Agent(ABC):
    def __init__(
        self,
        hyper_parameters: Dict,
        observation_space: Space,
        action_space: Space,
    ):
        self.hyper_parameters = hyper_parameters
        self.__set_input_space(observation_space)
        self.__set_output_space(action_space)

    @property
    @abstractmethod
    def hyper_parameters(self) -> Dict: ...

    @hyper_parameters.setter
    @abstractmethod
    def hyper_parameters(self, hyper_parameters: Dict): ...

    def __set_input_space(self, observation_space: Space): ...

    def __set_output_space(self, action_space: Space): ...

    @abstractmethod
    def get_action(self, *args, **kwargs): ...

    @abstractmethod
    def update(self, *args, **kwargs): ...

    @abstractmethod
    def save(self, *args, **kwargs): ...

    @abstractmethod
    def load(self, *args, **kwargs): ...
