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
    parser.add_argument("--gym-id", type=str, default="TaxiStrat-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-1,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--total-episodes", type=int, default=1000,
        help="total episodes of the experiments")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Log on wandb")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--video-freq", type=int, default=50,
        help="Frequency of saving videos, in episodes")    

    # Algorithm specific arguments
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--exploration-noise", type=float, default=0.5,
        help="the scale of exploration noise")
    parser.add_argument("--exploration-noise-end", type=float, default=0.01,
        help="the scale of exploration noise")
    # Arguments for DyLam
    parser.add_argument("--reward-scaling", type=float, default=1., help="reward scaling factor")
    parser.add_argument("--episodes-rb", type=int, default=10, help="number of episodes to calculate rb")
    parser.add_argument("--rew-tau", type=float, default=0.995, help="dylam tau")
    parser.add_argument("--dylam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Rather use DyLam or not")
    args = parser.parse_args()
    with open("dylam_hyperparameters.json", "r") as config_file:
        configs = json.load(config_file)
    configs = configs[args.gym_id]
    for key, value in configs.items():
        setattr(args, key, value)
    return args


class StratQLearning:
    def __init__(self, observation_space, action_space, hyper_params=None):
        self.action_size = action_space.n
        self.n_rewards = hyper_params.num_rewards
        self.observation_space = observation_space
        if isinstance(observation_space, gym.spaces.Discrete):
            self.obs_size = observation_space.n
            self.q_table = np.zeros((self.obs_size, self.action_size))
            self.components_q = np.zeros(
                (
                    self.n_rewards,
                    self.obs_size,
                    self.action_size,
                )
            )
        elif isinstance(observation_space, gym.spaces.Box):
            self.obs_size = observation_space.high - observation_space.low
            self.obs_size += 1
            self.q_table = np.zeros((*self.obs_size, self.action_size))
            self.components_q = np.zeros(
                (
                    self.n_rewards,
                    *self.obs_size,
                    self.action_size,
                )
            )

        self.lambdas = np.array(hyper_params.lambdas, dtype=np.float32)
        self.rb_reward = np.zeros((hyper_params.episodes_rb, self.n_rewards))
        self.rb_idx = 0
        self.can_dylam = False
        self.rew_tau = hyper_params.rew_tau
        self.rew_max = np.array(hyper_params.r_max, dtype=np.float32)
        self.rew_min = np.array(hyper_params.r_min, dtype=np.float32)
        self.last_rew_mean = None
        self.alpha = hyper_params.learning_rate
        self.gamma = hyper_params.gamma
        self.drQ = not hyper_params.dylam
        self.reward_scaling = hyper_params.reward_scaling

    def store_reward(self, reward):
        self.rb_reward[self.rb_idx] = reward
        self.rb_idx = (self.rb_idx + 1) % self.rb_reward.shape[0]
        self.can_dylam = self.rb_idx == 0 or self.can_dylam

    def dylam(self):
        if self.can_dylam and not self.drQ:
            rew_mean_t = np.mean(self.rb_reward, axis=0)
            if self.last_rew_mean is not None:
                rew_mean_t = (
                    rew_mean_t + (self.last_rew_mean - rew_mean_t) * self.rew_tau
                )
            dQ = np.clip(
                (self.rew_max - rew_mean_t) / (self.rew_max - self.rew_min), 0, 1
            )
            expdQ = np.exp(dQ) - 1
            self.lambdas = expdQ / (np.sum(expdQ, 0) + 1e-4)
            self.last_rew_mean = rew_mean_t

    def get_action(self, observation):
        if isinstance(self.observation_space, gym.spaces.Box):
            observation = tuple(observation)
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def update_component_tables(self, observation, action, reward, next_obs):
        if self.drQ:
            for i in range(self.n_rewards):
                update = reward[i] + self.gamma * (
                    self.components_q[i][next_obs].max()
                    - self.components_q[i][observation][action]
                )
                self.components_q[i][observation][action] = (
                    self.components_q[i][observation][action] + self.alpha * update
                )
        else:
            for i in range(self.n_rewards):
                update = reward[i] + self.gamma * (
                    self.components_q[i][next_obs].max()
                    - self.components_q[i][observation][action]
                )
                self.components_q[i][observation][action] = (
                    self.components_q[i][observation][action] + self.alpha * update
                )

    def update_policy(self, observation, action, reward, next_obs):
        try:
            if isinstance(self.observation_space, gym.spaces.Box):
                observation = tuple(observation)
                next_obs = tuple(next_obs)
            self.update_component_tables(observation, action, reward, next_obs)
            Qs = 0
            for i in range(self.n_rewards):
                Qs += self.lambdas[i] * self.components_q[i][observation][action]
            self.q_table[observation][action] = Qs
        except Exception:
            import ipdb

            ipdb.set_trace()


def main(args):
    strat_name = "DyLam" if args.dylam else "drQ"
    exp_name = f"Q_Learning_{strat_name}_{int(time.time())}_{args.gym_id}"
    project = "DyLam-Q"
    if args.seed == 0:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    args.method = "Q-DyLam" if args.dylam else "drQ"
    wandb.init(
        project=project,
        name=exp_name,
        entity="goncamateus",
        config=vars(args),
        monitor_gym=True,
        mode=None if args.track else "disabled",
        save_code=True,
        sync_tensorboard=True,
    )
    print(vars(args))
    writer = SummaryWriter(f"runs/{exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    env = mogym.make(args.gym_id, render_mode="rgb_array")
    if args.capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"monitor/{exp_name}",
            episode_trigger=lambda x: x % args.video_freq == 0,
        )
    agent = StratQLearning(
        env.observation_space,
        env.action_space,
        hyper_params=args,
    )
    epsilon = np.linspace(
        args.exploration_noise,
        args.exploration_noise_end,
        int(args.total_episodes * 0.7),
    )
    for episode in range(args.total_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        epi_reward = 0
        cumulative_original_reward = 0
        log = {}
        while not (done or truncated):
            if np.random.random() > epsilon[min(episode, len(epsilon) - 1)]:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            if "Original_reward" not in info:
                info["Original_reward"] = reward.sum()
            cumulative_original_reward += info["Original_reward"]
            epi_reward += reward
            if done:
                next_obs = obs
            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs
        log.update({f"ep_info/reward_total": cumulative_original_reward})
        writer.add_scalar("ep_info/total", cumulative_original_reward, episode)
        print(f"Episode {episode} reward: {epi_reward}")
        agent.store_reward(epi_reward)
        agent.dylam()
        for i in range(args.num_rewards):
            log.update({"lambdas/component_" + str(i): agent.lambdas[i].item()})
            writer.add_scalar(
                "lambdas/component_" + str(i), agent.lambdas[i].item(), episode
            )

        wandb.log(log)
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
