# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import argparse
import json
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

import envs
import wandb
from methods.sac_strat import SACStrat
from utils.experiment import StratSyncVectorEnv, make_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="LunarLanderOri-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1 - 5e-3,
        help="target smoothing coefficient (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--epsilon", type=float, default=1e-6,
            help="Epsilon to avoid zero denominator.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
 
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


def main(args):
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    strat_name = "DyLam" if args.dylam else "drQ"
    exp_name = f"SAC_{strat_name}_{int(time.time())}_{args.gym_id}"
    project = "DyLam"
    if args.seed == 0:
        args.seed = int(time.time())
    args.method = f"sac_{strat_name}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = StratSyncVectorEnv(
        [
            make_env(
                args,
                i,
                exp_name,
            )
            for i in range(args.num_envs)
        ],
        num_rewards=args.num_rewards,
    )

    agent = SACStrat(
        args,
        envs.single_observation_space,
        envs.single_action_space,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: training loop
    epi_rewards = np.zeros((args.num_envs, args.num_rewards))
    obs = envs.reset()
    log = {}
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put the logic for the algo here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(args.num_envs)]
            )
        else:
            actions = agent.get_action(obs)

        # TRY NOT TO MODIFY: execute the action and collect the next obs
        next_obs, rewards, dones, infos = envs.step(actions)
        epi_rewards = epi_rewards + rewards

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        agent.replay_buffer.add(obs, actions, rewards, next_obs, dones)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        for i, d in enumerate(dones):
            if d:
                agent.last_epi_rewards.add(epi_rewards[i])
                epi_rewards[i] = np.zeros(args.num_rewards)

        for item in infos:
            if "episode" in item.keys():
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

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # DyLam
            if args.dylam:
                lambdas = (
                    torch.ones(args.num_rewards).to(agent.device) / args.num_rewards
                )
            else:
                lambdas = torch.Tensor(args.lambdas).to(agent.device)
            r_max = torch.Tensor(args.r_max).to(agent.device)
            r_min = torch.Tensor(args.r_min).to(agent.device)
            rew_tau = args.rew_tau
            if agent.last_epi_rewards.can_do() and args.dylam:
                rew_mean_t = torch.Tensor(agent.last_epi_rewards.mean()).to(
                    agent.device
                )
                if agent.last_rew_mean is not None:
                    rew_mean_t = (
                        rew_mean_t + (agent.last_rew_mean - rew_mean_t) * rew_tau
                    )
                dQ = torch.clamp((r_max - rew_mean_t) / (r_max - r_min), 0, 1)
                expdQ = torch.exp(dQ) - 1
                lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
                agent.last_rew_mean = rew_mean_t

            update_actor = global_step % args.policy_frequency == 0
            policy_loss, qf1_loss, qf2_loss, alpha_loss = agent.update(
                args.batch_size, lambdas, update_actor
            )

            if global_step % args.target_network_frequency == 0:
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                for i in range(len(lambdas)):
                    log.update({"lambdas/component_" + str(i): lambdas[i].item()})
                    writer.add_scalar(
                        "lambdas/component_" + str(i), lambdas[i].item(), global_step
                    )
                log.update(
                    {
                        "losses/Value1_loss": qf1_loss.item(),
                        "losses/Value2_loss": qf2_loss.item(),
                        "losses/alpha": agent.alpha,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                )

                writer.add_scalar("losses/Value1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/Value2_loss", qf2_loss.item(), global_step)
                if update_actor:
                    log.update({"losses/policy_loss": policy_loss.item()})
                    writer.add_scalar(
                        "losses/policy_loss", policy_loss.item(), global_step
                    )
                writer.add_scalar("losses/alpha", agent.alpha, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    log.update({"losses/alpha_loss": alpha_loss.item()})
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
        wandb.log(log, global_step)
    agent.save(f"models/{exp_name}/")
    envs.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
