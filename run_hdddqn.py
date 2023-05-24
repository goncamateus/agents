import argparse
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
from methods.hdddqn import HDDDQN
from utils.experiment import make_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HierarchicalSimpleNav-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
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

    # Algorithm specific arguments
    parser.add_argument("--worker-max-episodes", type=int, default=100000,
        help="the number of episodes the worker runs")
    parser.add_argument("--manager-max-steps", type=int, default=50000,
        help="the number of steps the manager runs")
    parser.add_argument("--worker-batch-size", type=int, default=1024,
        help="the number of batches")
    parser.add_argument("--manager-batch-size", type=int, default=256,
        help="the number of batches")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--q-lr", type=float, default=0.0005,
        help="the learning rate of the Q network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=250000,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--manager-updates", type=int, default=3,
        help="the number of updates for each time the manager acts")
    parser.add_argument("--manager-target-update-freq", type=int, default=1000,
        help="the frequency of updates for the manager target")
    parser.add_argument("--worker-gamma", type=float, default=0.9,
        help="the discount factor gamma")
    parser.add_argument("--manager-gamma", type=float, default=1,
        help="the discount factor gamma")
    parser.add_argument("--eps-greedy-decay", type=float, default=0.4,
        help="the decay rate of epsilon greedy (0.1 at 100000 steps)")
    parser.add_argument("--pre-train-steps", type=int, default=150000,
        help="the number of pre-training steps before training manager")

    # PER parameters
    parser.add_argument("--alpha", type=float, default=0.6,
        help="determines how much prioritization is used")
    parser.add_argument("--beta", type=float, default=0.4,
        help="determines how much importance sampling is used")
    parser.add_argument("--prior-eps", type=float, default=1e-5,
            help="guarantees every transition can be sampled")
    # N-step Learning
    parser.add_argument("--num-steps", type=int, default=1,
        help="the step number to calculate n-step td error")
    args = parser.parse_args()
    args.hierarchical = True
    # fmt: on
    return args


def main(args):
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    exp_name = f"HDDDQN_{int(time.time())}_{args.gym_id}"
    # project = args.gym_id.split("-")[0]
    project = "DyLam"
    if args.seed == 0:
        args.seed = int(time.time())
    args.method = "hdddqn"
    wandb.init(
        project=project,
        name=exp_name,
        entity="goncamateus",
        config=vars(args),
        monitor_gym=True,
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
    env = make_env(args, 0, exp_name)()

    agent = HDDDQN(args, env.observation_space, env.action_space)
    start_time = time.time()

    # TRY NOT TO MODIFY: training loop
    global_step = 0
    obs = env.reset()
    log = {}
    info = {"worker": {"worker_done": False}}
    successes = {
        "worker": [],
        "manager": [],
    }
    resample_goal = False

    while agent.manager_action_count < args.manager_max_steps:
        agent.env_steps = env.steps_count
        # ALGO LOGIC: put the logic for the algo here
        if info["worker"]["worker_done"]:
            resample_goal = True
        action = agent.get_action(obs, global_step, resample_manager=resample_goal)

        # TRY NOT TO MODIFY: execute the action and collect the next obs
        next_obs, reward, done, info = env.step(action)
        transition = {
            "worker": [
                obs["worker"],
                action["worker"],
                reward["worker"],
                next_obs["worker"],
                info["worker"]["worker_done"],
            ],
            "manager": [
                obs["manager"],
                action["manager"],
                reward["manager"],
                next_obs["manager"],
                done,
            ],
        }
        agent.store_transition(transition, manager_store=resample_goal)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if info["worker"]["worker_done"]:
            log.update({"charts/epsilon_worker": agent.worker_epsilon})
            writer.add_scalar(
                "charts/epsilon_worker", agent.worker_epsilon, global_step
            )
            log.update(
                {
                    "charts/randomness_rate_worker": agent.randomness_rate_worker
                    / env.steps_count
                }
            )
            writer.add_scalar(
                "charts/randomness_rate_worker",
                agent.randomness_rate_worker / env.steps_count,
                global_step,
            )
            strat_rewards = [
                x for x in info["worker"].keys() if x.startswith("reward_")
            ]
            for key in strat_rewards:
                log.update(
                    {f"ep_info/{key.replace('reward_', '')}": info["worker"][key]}
                )
                writer.add_scalar(
                    f"charts/{key.replace('reward_', '')}",
                    info["worker"][key],
                    global_step,
                )
            successes["worker"].append(info["worker"]["reward_subobjective"])
            log.update(
                {"ep_info/worker_success_rate": np.mean(successes["worker"][-50:])}
            )
            writer.add_scalar(
                "charts/worker_success_rate",
                np.mean(successes["worker"][-50:]),
                global_step,
            )
            agent.randomness_rate_worker = 0
            agent.num_episodes["worker"] += 1

        if done:
            print(
                f'global_step={global_step}, Objective: {info["manager"]["reward_objective"]}'
            )
            log.update({f"ep_info/episodic_length": env.steps_count})
            writer.add_scalar("charts/episodic_length", env.steps_count, global_step)
            log.update({"charts/epsilon_manager": agent.manager_epsilon})
            writer.add_scalar(
                "charts/epsilon_manager", agent.manager_epsilon, global_step
            )
            strat_rewards = [
                x for x in info["manager"].keys() if x.startswith("reward_")
            ]
            for key in strat_rewards:
                log.update(
                    {f"ep_info/{key.replace('reward_', '')}": info["manager"][key]}
                )
                writer.add_scalar(
                    f"charts/{key.replace('reward_', '')}",
                    info["manager"][key],
                    global_step,
                )
            successes["manager"].append(info["manager"]["reward_objective"])
            log.update(
                {"ep_info/manager_success_rate": np.mean(successes["manager"][-50:])}
            )
            writer.add_scalar(
                "charts/manager_success_rate",
                np.mean(successes["manager"][-50:]),
                global_step,
            )

            agent.num_episodes["manager"] += 1
            done = False
            obs = env.reset()

        # ALGO LOGIC: training.
        model_logs = agent.update(global_step, writer, update_manager=resample_goal)
        log.update(model_logs)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        log.update({"charts/SPS": int(global_step / (time.time() - start_time))})
        wandb.log(log, global_step)
        global_step += 1
        resample_goal = False

    env.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
