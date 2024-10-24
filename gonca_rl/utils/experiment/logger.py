import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

#     logger.update_episode(episode_logs, global_step)
# logger.update_train(train_logs)
# logger.update_evaluation(evaluation_logs, global_step)
# logger.push()


class Logger(ABC):
    def __init__(self):
        self.__logs = dict()

    def __get_logs_from_info(
        self,
        info: dict,
        info_keys: Optional[list] = None,
        is_vector_env: bool = False,
        evaluation: bool = False,
    ) -> dict[str, float]:
        """Returns logs from info dictionary.

        Args:
            info (dict): Info dictionary from environment.
            info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.
            is_vector_env (bool, optional): Rather the environment is vectorized. Defaults to False.
            evaluation (bool, optional): Rather the episode is evaluation episode. Defaults to False.

        Returns:
            dict[str, float]: Dict of logs.
        """
        log_section = "Evaluation" if evaluation else "Episode"
        logs = dict()
        if info_keys is not None:
            for key in info_keys:
                if is_vector_env:
                    logs.update({f"{log_section}/{key}": np.mean(info[key])})
                else:
                    logs.update({f"{log_section}/{key}": info[key]})
        return logs

    def __get_logs_from_reward(
        self,
        episode_reward: float | np.ndarray,
        is_vector_env: bool = False,
        evaluation: bool = False,
    ) -> dict[str, float]:
        """Returns log from episode reward.

        Args:
            episode_reward (float | np.ndarray): Episode reward.
            is_vector_env (bool, optional): Rather the environment is vectorized. Defaults to False.
            evaluation (bool, optional): Rather the episode is evaluation episode. Defaults to False.

        Returns:
            dict[str, float]: Log dictionary from episode reward.
        """
        reward_log = episode_reward
        if is_vector_env:
            reward_log = np.mean(episode_reward)
        log_section = "Evaluation" if evaluation else "Episode"
        return {f"{log_section}/Return": reward_log}

    @abstractmethod
    def push(self):
        """Pushes logs to the logger."""
        raise NotImplementedError

    @abstractmethod
    def update_train(self, train_logs: Dict[str, float], global_step: int):
        """Updates train logs.

        Args:
            train_logs (Dict[str, float]): Train logs.
            global_step (int): Global step count.
        """
        self.__logs.update({f"Train/{key}": value for key, value in train_logs.items()})

    @abstractmethod
    def update_episode(
        self,
        episode_reward: float | np.ndarray,
        info: dict | np.ndarray,
        done: float | np.ndarray,
        truncated: float | np.ndarray,
        global_step: int,
        info_keys: Optional[list] = None,
        evaluation: bool = False,
    ) -> dict[str, float]:
        """Logs informations when the episode ends for one or multiple environments.

        Args:
            episode_reward (float | np.ndarray): Episode reward from the environment.
            info (dict | np.ndarray): Info dictionary from the environment.
            done (float | np.ndarray): Done flag from the environment.
            truncated (float | np.ndarray): Truncated flag from the environment.
            global_step (int): Global step count.
            info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.
            evaluation (bool, optional): Rather the episode is evaluation episode. Defaults to False.
        """
        _epi_reward = episode_reward
        _info = info
        is_vector_env = isinstance(episode_reward, np.ndarray)
        if is_vector_env:
            episode_ended = np.logical_or(done, truncated)
            _epi_reward = episode_reward[episode_ended]
            _info = {key: value[episode_ended] for key, value in info.items()}
        reward_log = self.__get_logs_from_reward(_epi_reward, is_vector_env, evaluation)
        info_logs = self.__get_logs_from_info(
            _info, info_keys, is_vector_env, evaluation
        )

        self.__logs.update(reward_log)
        self.__logs.update(info_logs)

    @abstractmethod
    def finish(self):
        """Finishes logging."""
        raise NotImplementedError
