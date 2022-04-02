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
from methods.ppo import PPO
from methods.ppo_strat import PPOStrat
from utils.experiment import StratSyncVectorEnv, make_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="LunarLanderStrat-v0",
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
    parser.add_argument("--num-rewards", type=int, default=10, help="number of rewards to alphas")
    parser.add_argument("--rew-tau", type=float, default=0.995, help="number of rewards to alphas")
    parser.add_argument("--dynamic-alphas", type=lambda x: bool(strtobool(x)), default=False, help="Rather use dynamic alphas or not")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


def main(args):
    exp_name = f"PPO_strat_{int(time.time())}_{args.gym_id}"
    print(vars(args))
    wandb.init(
        project="mestrado_ppo_lander",
        name=exp_name,
        entity="goncamateus",
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
    envs = StratSyncVectorEnv(
        [
            make_env(args.gym_id, args.seed + i, i, args.capture_video,
                     exp_name, statistics=False)
            for i in range(args.num_envs)
        ],
        num_rewards=args.num_rewards
    )
    agent = PPOStrat(args, envs).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, args.num_rewards)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, args.num_rewards)).to(device)
    epi_lenghts = np.zeros(args.num_envs)
    epi_rewards = np.zeros((args.num_envs, args.num_rewards))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_num = 0

    for update in tqdm(range(1, num_updates + 1)):
        log = {}
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            epi_lenghts = epi_lenghts + 1
            epi_rewards = epi_rewards + reward
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            
            for i in range(len(done)):
                if done[i]:
                    log.update({"ep_info/ep_steps": epi_lenghts[i]})
                    writer.add_scalar("ep_info/ep_steps", epi_lenghts[i], update)
                    epi_lenghts[i] = 0
                    for key, value in info[i].items():
                        if key != "terminal_observation":
                            log.update({f"ep_info/{key}": value})
                            writer.add_scalar(f"ep_info/{key}", value, update)
                    log.update({f"ep_info/ep_num": episode_num})        
                    writer.add_scalar(f"ep_info/ep_num", episode_num, update)
                    episode_num += 1
                    agent.last_epi_rewards.add(epi_rewards[i])
                    epi_rewards[i] = np.zeros(args.num_rewards)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            gamma = args.gamma
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = torch.zeros_like(next_value).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    bootstrapped = torch.zeros_like(nextvalues).to(device)
                    for i in range(len(nextnonterminal)):
                        bootstrapped[i] = gamma * nextnonterminal[i] * nextvalues[i]
                    delta = rewards[t] + bootstrapped- values[t]
                    last_gae_lm_advantages = torch.zeros_like(next_value).to(device)
                    for i in range(len(nextnonterminal)):
                        last_gae_lm_advantages[i] = gamma * args.gae_lambda * nextnonterminal[i] * lastgaelam[i]
                    advantages[t] = lastgaelam =  delta + last_gae_lm_advantages
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape((args.batch_size, args.num_rewards))
        b_returns = returns.reshape((args.batch_size, args.num_rewards))
        b_values = values.reshape((args.batch_size, args.num_rewards))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        alphas = torch.Tensor([0.35, 0, 0, 0, 0.1, 0, 0.002, 0.06, 0.06, 0.4]).to(
            agent.device
        )
        if args.dynamic_alphas:
            alphas = alphas / args.num_rewards
        r_max = torch.Tensor([0, 0, -0.03, -0.02, 0, -0.2, -0.2, 1, 1, 1]).to(
            agent.device
        )
        r_min = torch.Tensor([-1, -1, -0.8, -0.5, -1, -1, -1, -1, -1, -1]).to(
            agent.device
        )
        # Alpha automatic adjustment
        rew_tau = args.rew_tau
        if agent.last_epi_rewards.can_do() and args.dynamic_alphas:
            rew_mean_t = torch.Tensor(agent.last_epi_rewards.mean()).to(agent.device)
            if agent.last_rew_mean is not None:
                rew_mean_t = rew_mean_t + (agent.last_rew_mean - rew_mean_t) * rew_tau
            dQ = torch.clamp((r_max - rew_mean_t) / (r_max - r_min), 0, 1)
            expdQ = torch.exp(dQ) - 1
            alphas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
            agent.last_rew_mean = rew_mean_t

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                batch = (
                    b_obs[mb_inds],
                    b_logprobs[mb_inds],
                    b_actions[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                )
                old_approx_kl, approx_kl, v_loss, pg_loss, entropy_loss, alphas = agent.update(
                    clipfracs, batch, alphas
                )

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        log.update({"train/learning_rate": agent.optimizer.param_groups[0]["lr"]})
        writer.add_scalar(
            "train/learning_rate", agent.optimizer.param_groups[0]["lr"], update
        )
        for i in range(len(alphas)):
            log.update({"alphas/component_" + str(i): alphas[i].item()})
            writer.add_scalar(
                "alphas/component_" + str(i), alphas[i].item(), update
            )
        log.update({
            "losses/value_loss": agent.optimizer.param_groups[0]["lr"],
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "ep_info/SPS": int(global_step / (time.time() - start_time))
        })
        writer.add_scalar("losses/value_loss", v_loss.item(), update)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), update)
        writer.add_scalar("losses/entropy", entropy_loss.item(), update)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), update)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), update)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), update)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "ep_info/SPS", int(global_step / (time.time() - start_time)), update
        )
        wandb.log(log, step=update)
    envs.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
