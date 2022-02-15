import gym
import numpy as np
import pybullet_envs
import rsoccer_gym
import torch
from tqdm import tqdm

import wandb
from methods.ddpg import DDPGAgent
from utils.experiment import HyperParameters, save_checkpoint
from utils.wrappers import LunarLanderStratWrapper


def main():
    hp = HyperParameters(
        EXP_NAME="LunarLander-0",
        DEVICE=torch.device("cuda"),
        ENV_NAME="LunarLanderContinuous-v2",
        LEARNING_RATE=1e-3,
        BATCH_SIZE=256,
        GAMMA=0.99,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=25e3,
        SAVE_FREQUENCY=1000,
        TOTAL_GRAD_STEPS=1000000,
    )
    env = gym.make(hp.ENV_NAME)
    env = LunarLanderStratWrapper(env)
    env = gym.wrappers.RecordVideo(
        env, "./monitor/", step_trigger=lambda x: x % 100000 == 0
    )

    wandb.init(
        project="mestrado",
        name=hp.EXP_NAME,
        entity="goncamateus",
        config=hp.to_dict(),
        monitor_gym=True,
        # mode="disabled",
    )
    agent = DDPGAgent(hp)

    grad_steps = 0
    state = env.reset()
    done = False
    epi_reward = 0
    epi_steps = 0
    log_dict = {}
    for _ in tqdm(range(hp.TOTAL_GRAD_STEPS), smoothing=0.1):
        action = agent.act(state)
        action += 0.1 * np.random.randn(env.action_space.shape[0])
        action = np.clip(action, -env.action_space.high[0], env.action_space.high[0])
        action[0] = np.clip(action[0], 0.0, 1.0)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        epi_reward += reward
        epi_steps += 1

        if len(agent.replay_buffer) >= hp.REPLAY_INITIAL:
            # if epi_steps % 30 == 0:
            train_logs = agent.train()
            log_dict.update(train_logs)
            grad_steps += 1

        if grad_steps % hp.SAVE_FREQUENCY == 0 and grad_steps > 0:
            nets = {"actor": agent.actor, "critic": agent.critic}
            optims = {
                "actor_optimizer": agent.actor_optimizer,
                "critic_optimizer": agent.critic_optimizer,
            }
            save_checkpoint(hp, grad_steps, nets, optims)

        info = {
            "ep_info/" + key: item
            for key, item in info.items()
            if "truncated" not in key
        }
        log_dict.update(info)
        if done:
            log_dict["ep_info/ep_rw"] = epi_reward
            log_dict["ep_info/ep_steps"] = epi_steps

        wandb.log(log_dict)
        if done:
            state = env.reset()
            epi_reward = 0
            epi_steps = 0
            log_dict = {}


if __name__ == "__main__":
    main()
