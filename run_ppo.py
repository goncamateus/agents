# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
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
from methods.ppo import PPO
from methods.ppo_continuous import PPOContinuous
from utils.experiment import make_env


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
    parser.add_argument("--total-timesteps", type=int, default=25000,
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
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the number of batches")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--log-std-init", type=float, default=0,
        help="Initial value for log std when actions are continuous")
    args = parser.parse_args()
    args.buffer_size = int(args.num_envs * args.num_steps)
    args.n_batches = int(args.buffer_size // args.batch_size)
    # fmt: on
    return args


def main(args):
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    exp_name = f"PPO_{int(time.time())}_{args.gym_id}"
    # project = args.gym_id.split("-")[0]
    project = "DyLam"
    if args.seed == 0:
        args.seed = int(time.time())
    args.method = "ppo"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args,
                i,
                exp_name,
            )
            for i in range(args.num_envs)
        ],
    )

    algo = PPO if not args.continuous else PPOContinuous
    agent = algo(args, envs.single_observation_space, envs.single_action_space).to(
        device
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    num_updates = args.total_timesteps // args.buffer_size
    start_time = time.time()

    for update in range(1, num_updates + 1):
        log = {}
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            for item in info:
                if "episode" in item.keys():
                    print(
                        f'global_step={global_step}, episodic_return={item["Original_reward"]}'
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
        returns, advantages = agent.value_bootstrap(
            next_obs, next_done, rewards, dones, values
        )
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = np.random.permutation(args.buffer_size)
            for start in range(0, args.buffer_size, args.batch_size):
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                batch = (
                    b_obs[mb_inds],
                    b_logprobs[mb_inds],
                    b_actions[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                )
                old_approx_kl, approx_kl, v_loss, pg_loss, entropy_loss = agent.update(
                    clipfracs, batch
                )

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        log.update({"train/learning_rate": agent.optimizer.param_groups[0]["lr"]})
        log.update(
            {
                "losses/value_loss": agent.optimizer.param_groups[0]["lr"],
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "ep_info/SPS": int(global_step / (time.time() - start_time)),
            }
        )
        writer.add_scalar(
            "charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        wandb.log(log, step=global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
