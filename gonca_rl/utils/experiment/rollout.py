import gymnasium as gym
import numpy as np

from typing import Optional

from gonca_rl.agents import Agent
from gonca_rl.utils.experiment.logger import Logger


def update_agent(
    agent: Agent, global_step: int, logger: Logger
) -> dict[str, float] | None:
    """Updates agent if needed.

    Args:
        agent (Agent): Agent to update.
        global_step (int): Global step count.
        logger (Logger): Logger to update.

    Returns:
        dict[str, float] | None: List of logs when can log.
    """
    train_log = None
    if global_step % agent.update_interval() == 0:
        train_log = agent.update()
        logger.update_train(train_log, global_step)
    return train_log


def treats_next_state(
    state: np.ndarray,
    next_state: np.ndarray,
    done: bool | np.ndarray,
    is_vector_env: bool,
):
    """Treats next state when the episode ends.

    Args:
        state (np.ndarray): Current state.
        next_state (np.ndarray): Next state.
        done (bool | np.ndarray): Done flag.
        is_vector_env (bool): Rather the environment is vectorized.

    Returns:
        np.ndarray: Real next state
    """

    real_next_state = next_state
    if is_vector_env:
        for i in range(next_state):
            if done[i]:
                real_next_state[i] = state[i]
    else:
        if done:
            real_next_state = state
    return real_next_state


def evaluate(
    agent: Agent,
    env: gym.Env,
    logger: Logger,
    global_step: int,
    info_keys: Optional[list] = None,
):
    """Evaluates agent if needed.

    Args:
        agent (Agent): Agent to evaluate.
        env (gym.Env): Gym environment.
        logger (Logger): Logger to update.
        global_step (int): Global step count.
        info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.
    """
    eval_env = gym.make(env.spec.id)
    if global_step % agent.eval_interval() == 0:
        eval_reward = 0
        state, _ = eval_env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.get_action(state, is_training=False)
            state, reward, done, truncated, info = env.step(action)
            eval_reward += reward
        logger.update_episode(
            eval_reward, info, done, truncated, global_step, info_keys, evaluation=True
        )


def rollout(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    max_step_count: int,
    info_keys: Optional[list] = None,
    is_training: bool = True,
) -> tuple[dict[str, float], list[tuple[dict[str, float], int]]]:
    """Rollout for a single episode.
        This function is responsible for interacting with the environment for a single episode
        and updating the agent's memory when there is one.

    Args:
        env (gym.Env): gymnaisum environment.
        agent (Agent): Agent to interact with environment.
        logger (Logger): Logger to update.
        max_step_count (int): Maximum step count for the experiment.
        info_keys (Optional[list], optional): Keys from info dictionary that we want to log. Defaults to None.
        is_training (bool, optional): Rather the agent is training. Defaults to True.

    Returns:
        tuple[dict[str, float], list[tuple[dict[str, float], int]]]: tuple of episode logs and train logs.
    """
    is_vector_env = isinstance(env, gym.vector.VectorEnv)
    episode_reward = 0
    state, info = env.reset()
    global_step = 0
    while global_step < max_step_count:
        action = agent.get_action(state, is_training)
        next_state, reward, done, truncated, info = env.step(action)
        real_next_state = treats_next_state(state, next_state, done, is_vector_env)
        agent.memorize(state, action, reward, real_next_state, done)
        if is_training:
            update_agent(agent, global_step)
        episode_reward += reward
        state = next_state
        global_step += 1
        logger.update_episode(
            episode_reward, info, done, truncated, global_step, info_keys
        )
        evaluate(agent, env, logger, global_step, info_keys)
        logger.push()
    env.close()
    logger.finish()
