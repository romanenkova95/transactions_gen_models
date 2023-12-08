"""File with the definition of custom convolutions for TS2Vec."""
import torch
import torch.nn.functional as F
from torch import nn


class SamePadConv(nn.Module):
    """1D-convolution preserving sequence length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """Init SamePadConv.

        Args:
        ----
            in_channels (int): input size
            out_channels (int): output size
            kernel_size (int): convolutional kernel size
            dilation (int): dilation factor
            groups (int): controls the connections between inputs and outputs (stndard parameter of nn.Conv1d)
        """
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve input tensor.

        Args:
        ----
            x (torch.Tensor): input tensor

        Returns:
        -------
            conv(x)
        """
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    """Block of dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ) -> None:
        """Init ConvBlock.

        Args:
        ----
            in_channels (int): input size
            out_channels (int): output size
            kernel_size (int): convolutional kernel size
            dilation (int): dilation factor
            final (bool): indicates if the block is the last one
        """
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass throught the block.

        Args:
        ----
            x (torch.Tensor): input

        Returns:
        -------
            output of the block
        """
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    """Dilated 1D convolutional sequence encoder."""

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int) -> None:
        """Initialize DilatedConvEncoder.

        Args:
        ----
            in_channels (int): input size
            channels (List[int]): list of hidden sizes (num of channels for each convolution)
            kernel_size (int): kernel size
        """
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass throught the model.

        Args:
        ----
            x (torch.Tensor): input

        Returns:
        -------
            output = encoded input
        """
        return self.net(x)
