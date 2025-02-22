import torch
import torch.nn as nn


def xavier_init(m):
    """
    Applies Xavier initialization to the weights of a given layer if it is an instance of nn.Linear.

    This function initializes the weights of the layer using the Xavier uniform distribution and sets the biases to zero.

    Parameters:
    m (torch.nn.Module): The layer to initialize. If the layer is an instance of nn.Linear, its weights and biases will be initialized.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def kaiming_init(layer, bias_const=0.0):
    """
    Initializes the weights of a given layer using Kaiming Normal initialization
    and sets the bias to a constant value.

    Args:
        layer (torch.nn.Module): The layer to initialize.
        bias_const (float, optional): The constant value to initialize the bias.
                                      Default is 0.0.

    Returns:
        torch.nn.Module: The layer with initialized weights and bias.
    """
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def target_soft_update(target, source, tau):
    """
    Perform a soft update of the target network parameters using the source network parameters.

    Args:
        target (torch.nn.Module): The target network whose parameters will be updated.
        source (torch.nn.Module): The source network whose parameters will be used for the update.
        tau (float): The interpolation parameter, where 0 < tau < 1. A higher tau means the target network
                     parameters will be updated more towards the source network parameters.

    Returns:
        torch.nn.Module: The updated target network.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return target
