import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """
    A neural network model for Q-Values.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        hidden_dim (int, optional): Dimension of the hidden layers. Default is 256.
        back_bone_size (int, optional): Number of hidden layers in the backbone. Default is 2.
        number_outputs (int, optional): Number of output values. Default is 1.
        initializer (callable, optional): A function to initialize the network parameters. Default is None.

    Attributes:
        back_bone_size (int): Number of hidden layers in the backbone.
        input_layer (nn.Linear): The input layer of the network.
        back_bone (nn.ModuleList): A list of hidden layers in the backbone.
        output_layer (nn.Linear): The output layer of the network.

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
        super(QNet, self).__init__()
        self.back_bone_size = back_bone_size
        self.input_layer = nn.Linear((state_dim + action_dim), hidden_dim)
        self.back_bone = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(back_bone_size)
        ]
        self.back_bone = nn.ModuleList(self.back_bone)
        self.output_layer = nn.Linear(hidden_dim, number_outputs)

        if initializer is not None:
            self.apply(initializer)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = F.relu(self.input_layer(x))
        for layer in self.back_bone:
            x = F.relu(layer(x))
        return self.output_layer(x)
