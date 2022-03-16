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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="LunarLanderContinuous-v2",
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Log on wandb")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.995,
        help="Tau updates on neural networks")
    parser.add_argument("--replay-init", type=int, default=10000,
        help="Size of replay buffer before training")
    parser.add_argument("--replay-size", type=int, default=1000000,
        help="Max size of replay buffer")
    parser.add_argument("--batch-size", type=int, default=64,
        help="Batch size for training")
    parser.add_argument("--update-freq", type=int, default=4,
        help="Update every freq steps")
    parser.add_argument('--noise-theta', type=float, default=0.15,
        help='noise theta')
    parser.add_argument('--noise-sigma', type=float, default=0.2, 
        help='noise sigma') 
    parser.add_argument('--noise-mu', type=float, default=0.0,
        help='noise mu')
    parser.add_argument('--epsilon-decay', type=int, default=100000,
        help='linear decay of exploration policy') 
    args = parser.parse_args()

    return args


def main(args):
    exp_name = f"DDPG_{int(time.time())}_{args.gym_id}"
    print(vars(args))
    wandb.init(
        project="mestrado_ddpg_lander",
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
            make_env(args.gym_id, args.seed + i, i, args.capture_video, exp_name)
            for i in range(args.num_envs)
        ]
    )
    agent = DDPGAgent(args, envs).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps
    obs = envs.reset()
    epi_lenghts = np.zeros(args.num_envs)
    epi_rewards = np.zeros(args.num_envs)

    for update in tqdm(range(1, num_updates + 1)):
        global_step += 1 * args.num_envs
        actions = agent.act(obs)
        next_obs, reward, done, info = envs.step(actions)
        epi_lenghts = epi_lenghts + 1
        epi_rewards = epi_rewards + reward
        for i in range(args.num_envs):
            if done[i]:
                writer.add_scalar("ep_info/ep_steps", epi_lenghts[i], global_step)
                writer.add_scalar("ep_info/Original_reward", epi_rewards[i], global_step)
                terminal_obs = info[i]["terminal_observation"]
                agent.observe(obs[i], actions[i], reward[i], terminal_obs, done[i])
                epi_lenghts[i] = 0
                epi_rewards[i] = 0
            else:
                agent.observe(obs[i], actions[i], reward[i], next_obs[i], done[i])
            obs[i] = next_obs[i]

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
