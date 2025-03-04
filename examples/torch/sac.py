# CleanRL SAC script using my implementations
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from agents.torch.architectures.utils import target_soft_update
from agents.torch.policy_gradient.sac import TorchSAC as SAC


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_frequency: int = 25
    """record video every n episodes"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    hidden_dim: int = 256
    """the dimension for the hidden layers"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    reward_scale: float = 1.0
    """the value scaling of the rewards"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def get_hyperparameters(args, envs):
    action_range = (envs.single_action_space.high, envs.single_action_space.low)
    return {
        "gamma": args.gamma,
        "device": "cuda" if args.cuda else "cpu",
        "reward_scale": args.reward_scale,
        "buffer_size": args.buffer_size,
        "hidden_dim": args.hidden_dim,
        "q_learning_rate": args.q_lr,
        "policy_learning_rate": args.policy_lr,
        "alpha": args.alpha,
        "automatic_entropy_tuning": args.autotune,
        "action_range": action_range,
    }


def make_env(env_id, seed, idx, capture_video, run_name, video_frequency):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % video_frequency == 0,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
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
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                args.video_frequency,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.single_action_space.high[0])
    hyper_parameters = get_hyperparameters(args, envs)
    agent = SAC(
        hyper_parameters=hyper_parameters,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = agent.get_action(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for i in range(len(infos["episode"]["r"])):
                if terminations[i] or truncations[i]:
                    print(
                        f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", infos["episode"]["r"][i], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", infos["episode"]["l"][i], global_step
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = obs[idx]
        agent.replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            update_actor = global_step % args.policy_frequency == 0
            policy_loss, qf1_loss, qf2_loss, alpha_loss = agent.update(
                args.batch_size, update_actor=update_actor
            )
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                agent.target_critic = target_soft_update(
                    agent.critic, agent.target_critic, tau=args.tau
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", policy_loss.item(), global_step)
                writer.add_scalar("losses/alpha", agent.alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()
    writer.close()
