# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import argparse
import json
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

import envs
import wandb

from methods.dqn import DQNAgent


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

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--epsilon", type=float, default=1.,
        help="epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.07,
        help="epsilon end")
    parser.add_argument("--epsilon-decay", type=float, default=0.9999,
        help="epsilon decay")
    parser.add_argument("--num-epochs", type=int, default=8,
        help="the number of epochs to update the policy")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-update-freq", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
 
    # Arguments for DyLam
    parser.add_argument("--reward-scaling", type=float, default=1., help="reward scaling factor")
    parser.add_argument("--episodes-rb", type=int, default=10, help="number of episodes to calculate rb")
    parser.add_argument("--rew-tau", type=float, default=0.995, help="dylam tau")
    parser.add_argument("--dylam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Rather use DyLam or not")
    parser.add_argument("--stratified", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Rather use stratified rewards or not")

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
    if not args.stratified:
        method_name = "DQN"
    else:
        if args.dylam:
            method_name = "DQN_DyLam"
        else:
            method_name = "DQN_drQ"
    exp_name = f"{method_name}_{int(time.time())}_{args.gym_id}"
    project = "DyLam"
    if args.seed == 0:
        args.seed = int(time.time())
    args.method = method_name.split("_")[1] if args.stratified else method_name
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
    try:
        env = gym.make(args.gym_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        env = mo_gym.make(args.gym_id, render_mode="rgb_array")
    if args.capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"monitor/{exp_name}",
            episode_trigger=lambda x: x % args.video_freq == 0,
        )

    agent = DQNAgent(
        args,
        env.observation_space,
        env.action_space,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: training loop
    epi_reward = np.zeros(args.num_rewards)
    obs, _ = env.reset()
    log = {}
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put the logic for the algo here
        if global_step < args.batch_size:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)

        # TRY NOT TO MODIFY: execute the action and collect the next obs
        next_obs, rewards, done, truncated, info = env.step(action)
        if not args.stratified:
            if isinstance(rewards, np.ndarray):
                rewards = rewards.sum()
        epi_reward = epi_reward + rewards

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        agent.replay_buffer.add([obs], [action], [rewards], [next_obs], [done])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if done or truncated:
            agent.rb_rewards.add(epi_reward)
            agent.dylam()
            if "Original_reward" in info:
                print(
                    f"global_step={global_step}, episodic_return={info['Original_reward']}"
                )
                log.update({f"ep_info/reward_total": info["Original_reward"]})
                writer.add_scalar(
                    "charts/episodic_return", info["Original_reward"], global_step
                )
            else:
                print(
                    f"global_step={global_step}, episodic_return={(epi_reward * args.lambdas)[0]}"
                )
                log.update({f"ep_info/reward_total": (epi_reward * args.lambdas)[0]})
                writer.add_scalar(
                    "charts/episodic_return",
                    (epi_reward * args.lambdas)[0],
                    global_step,
                )
            epi_reward = np.zeros(args.num_rewards)
            strat_rewards = [x for x in info.keys() if x.startswith("reward_")]
            for key in strat_rewards:
                log.update({f"ep_info/{key.replace('reward_', '')}": info[key]})
                writer.add_scalar(
                    f"charts/{key.replace('reward_', '')}", info[key], global_step
                )
            obs, _ = env.reset()

        # ALGO LOGIC: training.
        if global_step > args.batch_size:
            update_policy = global_step % args.policy_frequency == 0
            if update_policy:
                policy_loss, component_loss = agent.update(args.batch_size)

            if global_step % 100 == 0:
                for i in range(len(agent.lambdas)):
                    log.update({"lambdas/component_" + str(i): agent.lambdas[i]})
                    writer.add_scalar(
                        "lambdas/component_" + str(i),
                        agent.lambdas[i],
                        global_step,
                    )
                if update_policy:
                    log.update({"losses/policy_loss": policy_loss})
                    writer.add_scalar("losses/policy_loss", policy_loss, global_step)
                    if args.stratified:
                        for i in range(len(component_loss)):
                            log.update(
                                {"losses/component_" + str(i): component_loss[i]}
                            )
                            writer.add_scalar(
                                "losses/component_" + str(i),
                                component_loss[i],
                                global_step,
                            )
                log.update(
                    {
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

        wandb.log(log, global_step)
    agent.save(f"models/{exp_name}/")
    env.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
