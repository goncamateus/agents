import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) policy network for reinforcement learning.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        hidden_dim (int, optional): Dimension of the hidden layers. Default is 256.
        back_bone_size (int, optional): Number of hidden layers in the backbone. Default is 2.
        initializer (callable, optional): Initialization function for the network parameters. Default is None.

    Attributes:
        back_bone_size (int): Number of hidden layers in the backbone.
        input_layer (nn.Linear): Linear layer for the input state.
        back_bone (nn.ModuleList): List of hidden layers in the backbone.
        output_layer (nn.Linear): Linear layer for the output action.

    Methods:
        forward(observation):
            Forward pass through the network.

            Args:
                observation (torch.Tensor): Input state tensor.

            Returns:
                torch.Tensor: Output action tensor.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        back_bone_size: int = 2,
        initializer=None,
    ):
        super(MLPPolicy, self).__init__()
        self.back_bone_size = back_bone_size
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.back_bone = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(back_bone_size)
        ]
        self.back_bone = nn.Sequential(*self.back_bone)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        if initializer is not None:
            self.apply(initializer)

    def forward(self, observation):
        x = torch.relu(self.input_layer(observation))
        for layer in self.back_bone:
            x = torch.relu(layer(x))
        return self.output_layer(x)
