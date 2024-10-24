import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    def __init__(self): ...

    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]: ...

    @abstractmethod
    def memorize(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ): ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def load(self, path: str): ...

    @abstractmethod
    def update_interval(self) -> int: ...
