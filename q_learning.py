import argparse
import json
import os
import time
from distutils.util import strtobool

import gymnasium as gym
import mo_gymnasium as mogym
import numpy as np
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

import envs
import wandb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Taxi-v3",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-1,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--total-episodes", type=int, default=1000,
        help="total episodes of the experiments")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Log on wandb")

    # Algorithm specific arguments
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--exploration-noise", type=float, default=0.5,
        help="the scale of exploration noise")
    parser.add_argument("--exploration-noise-end", type=float, default=0.01,
        help="the scale of exploration noise")
    args = parser.parse_args()
    with open("dylam_hyperparameters.json", "r") as config_file:
        configs = json.load(config_file)
    if args.gym_id in configs:
        configs = configs[args.gym_id]
        for key, value in configs.items():
            setattr(args, key, value)
    return args


class QLearningAgent:
    def __init__(self, observation_space, action_space, hyper_params=None):
        self.action_size = action_space.n
        self.observation_space = observation_space
        if isinstance(observation_space, gym.spaces.Discrete):
            self.obs_size = observation_space.n
            self.q_table = np.zeros((self.obs_size, self.action_size))
        elif isinstance(observation_space, gym.spaces.Box):
            self.obs_size = observation_space.high - observation_space.low
            self.obs_size += 1
            self.q_table = np.zeros((*self.obs_size, self.action_size))
        self.alpha = hyper_params.learning_rate
        self.gamma = hyper_params.gamma

    def get_action(self, observation):
        if isinstance(self.observation_space, gym.spaces.Box):
            observation = tuple(observation)
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def update_policy(self, observation, action, reward, next_obs):
        if isinstance(self.observation_space, gym.spaces.Box):
            observation = tuple(observation)
            next_obs = tuple(next_obs)
        update = reward + self.gamma * (
            self.q_table[next_obs].max() - self.q_table[observation][action]
        )
        self.q_table[observation][action] = (
            self.q_table[observation][action] + self.alpha * update
        )


def main(args):
    exp_name = f"Q_Learning_{int(time.time())}_{args.gym_id}"
    project = "DyLam"
    if args.seed == 0:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    args.method = "Q-Learning"
    wandb.init(
        project=project,
        name=exp_name,
        entity="goncamateus",
        config=vars(args),
        monitor_gym=False,
        mode=None if args.track else "disabled",
        save_code=True,
    )
    print(vars(args))
    writer = SummaryWriter(f"runs/{exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    env = mogym.make(args.gym_id, render_mode="ansi")
    agent = QLearningAgent(env.observation_space, env.action_space, hyper_params=args)
    epsilon = np.linspace(
        args.exploration_noise,
        args.exploration_noise_end,
        int(args.total_episodes * 0.7),
    )
    for episode in range(args.total_episodes):
        obs, info = env.reset()
        done = False
        epi_reward = 0
        log = {}
        while not done:
            if np.random.random() > epsilon[min(episode, len(epsilon) - 1)]:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            if isinstance(reward, np.ndarray):
                reward = reward.sum()
            epi_reward += reward
            if done:
                next_obs = obs
            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs
        print(f"Episode {episode} reward: {epi_reward}")
        log.update({f"ep_info/reward_total": epi_reward})
        wandb.log(log)

    env.close()

    print("---------------Evaluating---------------")
    env = mogym.make(args.gym_id, render_mode="human")
    for episodes in range(10):
        obs, info = env.reset()
        done = False
        truncated = False
        epi_reward = 0
        while not (done or truncated):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            if isinstance(reward, np.ndarray):
                reward = reward.sum()
            epi_reward += reward
        print(f"Evaluation Episode {episodes} reward: {epi_reward}")
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
