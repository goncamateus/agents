import gym
import rsoccer_gym
import torch
import wandb
from tqdm import tqdm

from methods.ddpg import DDPGAgent
from utils.experiment import HyperParameters, save_checkpoint
from utils.noise import OrnsteinUhlenbeckNoise as OUNoise


def schedule(episode_id):
    if episode_id % 80 == 0:
        return True


def main():
    hp = HyperParameters(
        EXP_NAME="DDPG-v0",
        DEVICE=torch.device("cuda"),
        ENV_NAME="VSS-v0",
        LEARNING_RATE=1e-3,
        BATCH_SIZE=256,
        GAMMA=0.95,
        NOISE_SIGMA_INITIAL=0.8,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=500000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        TOTAL_GRAD_STEPS=1000000,
    )
    env = gym.wrappers.Monitor(
        gym.make(hp.ENV_NAME), "./monitor/", force=True, video_callable=schedule
    )
    wandb.init(
        project="reward_alphas",
        name=hp.EXP_NAME,
        entity="robocin",
        config=hp.to_dict(),
        monitor_gym=True,
        mode=None,
        # "disabled",
    )
    agent = DDPGAgent(hp)
    noise = OUNoise(
        sigma=hp.NOISE_SIGMA_INITIAL,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high,
    )

    noise.reset()
    state = env.reset()
    steps = 0

    epi_reward = 0
    for i in tqdm(range(1, hp.TOTAL_GRAD_STEPS + hp.REPLAY_INITIAL), smoothing=0.1):
        log_dict = {}
        action = agent.act(state)
        action = noise(action)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        epi_reward += reward

        if done:
            info = {
                "ep_info/" + key: item
                for key, item in info.items()
                if "truncated" not in key
            }
            log_dict.update(info)
            log_dict["ep_info/ep_rw"] = epi_reward
            noise.reset()
            state = env.reset()
            epi_reward = 0

        if len(agent.replay_buffer) >= hp.REPLAY_INITIAL and i % 10 == 0:
            train_logs = agent.train()
            log_dict.update(train_logs)
            steps += 1
            wandb.log(log_dict)

        if (
            hp.NOISE_SIGMA_DECAY
            and noise.sigma > hp.NOISE_SIGMA_MIN
            and steps % hp.NOISE_SIGMA_GRAD_STEPS == 0
        ):
            noise.sigma *= hp.NOISE_SIGMA_DECAY

        if steps % hp.SAVE_FREQUENCY == 0:
            nets = {"actor": agent.actor, "critic": agent.critic}
            optims = {
                "actor_optimizer": agent.actor_optimizer,
                "critic_optimizer": agent.critic_optimizer,
            }
            save_checkpoint(hp, steps, nets, optims)


if __name__ == "__main__":
    main()
