import dataclasses

import gym
import rsoccer_gym
import torch
import os

import wandb


@dataclasses.dataclass
class HyperParameters:
    """Class containing all experiment hyperparameters"""

    EXP_NAME: str
    ENV_NAME: str
    LEARNING_RATE: float
    REPLAY_SIZE: int  # Maximum Replay Buffer Sizer
    REPLAY_INITIAL: int  # Minimum experience buffer size to start training
    SAVE_FREQUENCY: int  # Save checkpoint every _ grad_steps
    BATCH_SIZE: int
    GAMMA: float  # Reward Decay
    MAX_EPISODE_STEPS: int = None
    N_OBS: int = None
    N_ACTS: int = None
    SAVE_PATH: str = None
    DEVICE: str = None
    TOTAL_GRAD_STEPS: int = None
    NOISE_SIGMA_INITIAL: float = None,
    NOISE_THETA: float = None,
    NOISE_SIGMA_DECAY: float = None,
    NOISE_SIGMA_MIN: float = None,
    NOISE_SIGMA_GRAD_STEPS: int = None,

    def to_dict(self):
        return self.__dict__

    def __post_init__(self):
        env = gym.make(self.ENV_NAME)
        self.N_OBS, self.N_ACTS= (
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.SAVE_PATH = os.path.join("saves", self.ENV_NAME, self.EXP_NAME)
        self.CHECKPOINT_PATH = os.path.join(self.SAVE_PATH, "checkpoints")
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        self.action_space = env.action_space
        self.observation_space = env.observation_space


def save_checkpoint(hp, steps, neural_nets, optimizers):
    checkpoint = dataclasses.asdict(hp)
    for name, net in neural_nets.items():
        checkpoint[name] = net.state_dict()
    for name, optim in optimizers.items():
        checkpoint[name] = optim.state_dict()
    filename = os.path.join(
        hp.CHECKPOINT_PATH, "checkpoint_{:09}.pth".format(steps)
    )
    torch.save(checkpoint, filename)

def soft_update(model, target, tau):
    """
    Blend params of target net with params from the model
    :param tau:
    """
    assert isinstance(tau, float)
    assert 0.0 < tau <= 1.0
    state = model.state_dict()
    tgt_state = target.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * tau + (1 - tau) * v
    target.load_state_dict(tgt_state)
