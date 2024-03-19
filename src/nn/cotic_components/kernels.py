"""Module with cotic kernels."""

import torch
from torch import nn


class Kernel(nn.Module):
    """MLP Kernel, takes x of shape (*, in_channels), returns kernel values of shape (*, in_channels, out_channels)."""

    def __init__(
        self,
        hidden1: int,
        hidden2: int,
        hidden3: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize Kernel network.

        Args:
        ----
            hidden1 (int): 1st hidden layer size
            hidden2 (int): 2nd hidden layer size
            hidden3 (int): 3rd hidden layer size
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.args = [hidden1, hidden2, hidden3, in_channels, out_channels]

        self.layer_1 = nn.Linear(in_channels, hidden1)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(hidden2, hidden3)
        self.relu_3 = nn.ReLU()
        self.layer_4 = nn.Linear(hidden3, in_channels * out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def recreate(self, in_channels: int) -> nn.Module:
        """Copy kernel network.

        Args:
        ----
            in_channels (int): number of input channels in a copied kernel

        Returns:
        -------
            identical kernel network with a new number of input channels
        """
        args = self.args.copy()
        args[3] = in_channels
        return type(self)(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwsrd pass of a network.

        Args:
        ----
            x (torch.Tensor): input tensor

        Returns:
        -------
            out (torch.Tensor) - output tensor
        """
        shape = list(x.shape)[:-1]
        shape += [self.in_channels, self.out_channels]
        x = self.relu_1(self.layer_1(x))
        x = self.relu_2(self.layer_2(x))
        x = self.relu_3(self.layer_3(x))
        out = self.layer_4(x)
        out = out.reshape(*shape)

        return out
