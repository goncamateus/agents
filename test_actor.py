import gym
import numpy as np
import rsoccer_gym
import torch

from pyvirtualdisplay import Display

from utils.wrappers import RepeatActionWrapper
from methods.sac import GaussianPolicy


def load_model(model_path):
    model = torch.load(model_path)
    return model


def main():
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    env = gym.make("SSLPathPlanning-v1")
    env = gym.wrappers.RecordVideo(
        env,
        f"monitor/caps",
        episode_trigger=lambda x: True,
    )
    env = RepeatActionWrapper(env, n_actions=1)
    state_dict = load_model("artifacts/model:v1/actor.pt")
    num_inputs = np.array(env.observation_space.shape).prod()
    num_actions = np.array(env.action_space.shape).prod()
    actor = GaussianPolicy(
        num_inputs,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        epsilon=1e-6,
        action_space=env.action_space,
    )
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.to("cpu")

    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            state = torch.Tensor(obs.reshape(1, -1))
            action = actor.get_action(state)[0]
            obs, reward, done, info = env.step(action)
        print(info)

    env.close()


if __name__ == "__main__":
    main()
