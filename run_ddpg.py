# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs
import wandb
from methods.ddpg import DDPGAgent
from utils.experiment import make_env
from utils.noise import OrnsteinUhlenbeckNoise


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Pendulum-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=3000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--video-freq", type=int, default=50,
        help="Frequency of saving videos, in episodes")    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Log on wandb")
    parser.add_argument("--continuous", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to use continuous actions")
    parser.add_argument("--normalize", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to Normalize observations and rewards")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.999,
        help="Tau updates on neural networks")
    parser.add_argument("--replay-init", type=int, default=10000,
        help="Size of replay buffer before training")
    parser.add_argument("--replay-size", type=int, default=1000000,
        help="Max size of replay buffer")
    parser.add_argument("--batch-size", type=int, default=256,
        help="Batch size for training")
    parser.add_argument("--update-freq", type=int, default=20,
        help="Update every freq steps")
    parser.add_argument('--noise-theta', type=float, default=0.15,
        help='noise theta')
    parser.add_argument('--noise-sigma', type=float, default=0.2, 
        help='noise sigma') 
    parser.add_argument('--noise-mu', type=float, default=0.0,
        help='noise mu')
    args = parser.parse_args()

    return args


def main(args):
    exp_name = f"DDPG_{int(time.time())}_{args.gym_id}"
    print(vars(args))
    wandb.init(
        project="Mujoco",
        name=exp_name,
        entity="goncamateus",
        sync_tensorboard=True,
        config=vars(args),
        monitor_gym=True,
        mode=None if args.track else "disabled",
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args, i, exp_name)
            for i in range(args.num_envs)
        ]
    )
    agent = DDPGAgent(args, envs).to(device)
    noise = OrnsteinUhlenbeckNoise(
        sigma=args.noise_sigma,
        theta=args.noise_theta,
        min_value=envs.single_action_space.low,
        max_value=envs.single_action_space.high
    )
    noise.reset()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps
    obs = envs.reset()
    epi_lenghts = np.zeros(args.num_envs)
    epi_rewards = np.zeros(args.num_envs)
    log = {}

    for update in range(1, num_updates + 1):
        global_step += 1 * args.num_envs
        actions = agent.act(obs)
        actions = noise(actions)
        next_obs, reward, done, info = envs.step(actions)
        epi_lenghts = epi_lenghts + 1
        epi_rewards = epi_rewards + reward
        for item in info:
            if "episode" in item.keys():
                noise.reset()
                print(
                    f"global_step={global_step}, episodic_return={item['Original_reward']}"
                )
                log.update({f"ep_info/reward_total": item["Original_reward"]})
                writer.add_scalar(
                    "charts/episodic_return", item["Original_reward"], global_step
                )
                log.update({f"ep_info/episodic_length": item["episode"]["l"]})
                writer.add_scalar(
                    "charts/episodic_length", item["episode"]["l"], global_step
                )
                strat_rewards = [x for x in item.keys() if x.startswith("reward_")]
                for key in strat_rewards:
                    log.update({f"ep_info/{key.replace('reward_', '')}": item[key]})
                    writer.add_scalar(
                        f"charts/{key.replace('reward_', '')}", item[key], global_step
                    )
                break
        agent.observe(obs, actions, reward, next_obs, done)
        obs = next_obs

        if len(agent.replay_buffer) >= args.replay_init:
            agent.training = True
            if update % args.update_freq == 0:
                train_logs = agent.train()
                writer.add_scalar("losses/value_loss", train_logs['loss_Q'], global_step)
                writer.add_scalar("losses/policy_loss", train_logs['loss_pi'], global_step)
        writer.add_scalar(
            "ep_info/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
