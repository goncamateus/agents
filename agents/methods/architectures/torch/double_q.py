import torch.nn as nn

from agents.methods.architectures.torch.q_net import QNet


class DoubleQNet(nn.Module):
    """
    A neural network model for Double Q-learning.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        hidden_dim (int, optional): Dimension of the hidden layers. Default is 256.
        back_bone_size (int, optional): Number of hidden layers in the backbone. Default is 2.
        initializer (callable, optional): A function to initialize the network parameters. Default is None.

    Attributes:
        q_net1 (QNet): The first Q network.
        q_net2 (QNet): The second Q network.

    Methods:
        forward(x):
            Defines the forward pass of the network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        back_bone_size: int = 2,
        number_outputs: int = 1,
        initializer=None,
    ):
        super(DoubleQNet, self).__init__()
        self.q_net1 = QNet(
            state_dim,
            action_dim,
            hidden_dim,
            back_bone_size,
            number_outputs,
            initializer,
        )
        self.q_net2 = QNet(
            state_dim,
            action_dim,
            hidden_dim,
            back_bone_size,
            number_outputs,
            initializer,
        )

    def forward(self, observation, action):
        return self.q_net1(observation, action), self.q_net2(observation, action)
