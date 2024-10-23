import gymnasium as gym
import numpy as np

from gonca_rl.agents import Agent
from typing import Optional


def end_of_episode(done: bool, truncated: bool) -> bool:
    return done or truncated


def get_logs_from_info(
    info: dict, info_keys: Optional[list] = None, is_vector_env: bool = False
) -> list[dict[str, float]]:
    logs = []
    if info_keys is not None:
        for key in info_keys:
            if is_vector_env:
                logs.append({key: np.mean(info[key])})
            else:
                logs.append({key: info[key]})
    return logs


def get_log_from_reward(
    episode_reward: float | np.ndarray, is_vector_env: bool = False
) -> dict[str, float]:
    reward_log = episode_reward
    if is_vector_env:
        reward_log = np.mean(episode_reward)
    return {"Episode/Return": reward_log}


def episode_rollout(
    env: gym.Env, agent: Agent, info_keys: Optional[list] = None
) -> tuple[float, list[dict[str, float]]]:
    state = env.reset()
    done, truncated = False, False
    episode_reward = 0
    is_vector_env = isinstance(env, gym.vector.VectorEnv)
    while not end_of_episode(done, truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        if is_vector_env:
            done = np.all(done)
            truncated = np.all(truncated)
        agent.memorize(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
    logs = [get_log_from_reward(episode_reward, is_vector_env)]
    logs += get_logs_from_info(info, info_keys, is_vector_env)
    return episode_reward, logs
