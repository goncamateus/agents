import gymnasium as gym
import numpy as np

from typing import Optional

from gonca_rl.agents import Agent
from gonca_rl.utils.experiment.logger import Logger


def end_of_episode(done: bool, truncated: bool) -> bool:
    """Returns if the episode ended.

    Args:
        done (bool): Episode ended flag.
        truncated (bool): Epsiode max step reached flag.

    Returns:
        bool: Episode ended flag.
    """
    return done or truncated


def get_logs_from_info(
    info: dict, info_keys: Optional[list] = None, is_vector_env: bool = False
) -> list[dict[str, float]]:
    """Returns logs from info dictionary.

    Args:
        info (dict): Info dictionary from environment.
        info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.
        is_vector_env (bool, optional): Rather the environment is vectorized. Defaults to False.

    Returns:
        list[dict[str, float]]: List of logs.
    """
    logs = []
    if info_keys is not None:
        for key in info_keys:
            if is_vector_env:
                logs.append({f"Episode/{key}": np.mean(info[key])})
            else:
                logs.append({f"Episode/{key}": info[key]})
    return logs


def get_log_from_reward(
    episode_reward: float | np.ndarray, is_vector_env: bool = False
) -> dict[str, float]:
    """Returns log from episode reward.

    Args:
        episode_reward (float | np.ndarray): Episode reward.
        is_vector_env (bool, optional): Rather the environment is vectorized. Defaults to False.

    Returns:
        dict[str, float]: Log dictionary from episode reward.
    """
    reward_log = episode_reward
    if is_vector_env:
        reward_log = np.mean(episode_reward)
    return {"Episode/Return": reward_log}


def episode_rollout(
    env: gym.Env, agent: Agent, global_step: int, info_keys: Optional[list] = None
) -> list[dict[str, float]]:
    """Rollout for a single episode.
        This function is responsible for interacting with the environment for a single episode
        and updating the agent's memory when there is one.
    Args:
        env (gym.Env): gymnaisum environment.
        agent (Agent): Agent to interact with environment.
        global_step (int): Global step count from outer loop.
        info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.

    Returns:
        list[dict[str, float]]: List of logs.
    """
    is_vector_env = isinstance(env, gym.vector.VectorEnv)
    episode_reward = 0
    done, truncated = False, False
    logs = []
    
    state = env.reset()
    while not end_of_episode(done, truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        if is_vector_env:
            done = np.all(done)
            truncated = np.all(truncated)
        agent.memorize(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        global_step += 1
    logs.append(get_log_from_reward(episode_reward, is_vector_env))
    logs += get_logs_from_info(info, info_keys, is_vector_env)
    return logs


def train_rollout(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    max_step_count: int,
    info_keys: Optional[list] = None,
):
    global_step = 0
    while global_step < max_step_count:
        logs = episode_rollout(env, agent, global_step, info_keys)

