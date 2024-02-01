"""Code from the COTIC repository: https://github.com/VladislavZh/COTIC/tree/main."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class ContConv1d(nn.Module):
    """Continuous convolution layer for true events."""

    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        include_zero_lag: bool = False,
    ) -> None:
        """Initialize Continuous convolution layer.

        Args:
        ----
            kernel (nn.Module): Kernel neural net that takes (*,1) as input and returns (*, in_channles, out_channels) as output
            kernel_size (int): convolution layer kernel size
            in_channels (int): features input size
            out_channels (int): output size
            dilation (int): convolutional layer dilation (default = 1)
            include_zero_lag (bool): indicates if the model should use current time step features for prediction
        """
        super().__init__()
        assert dilation >= 1, "Wrong dilation size."
        assert in_channels >= 1, "Wrong in_channels."
        assert out_channels >= 1, "Wrong out_channels."

        self.kernel = kernel
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_zero_lag = include_zero_lag
        self.skip_connection = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / self.in_channels)
                for i in range(self.in_channels)
            ]
        )

        self.norm = nn.BatchNorm1d(out_channels)

    def __temporal_enc(self, time: torch.Tensor) -> torch.Tensor:
        """Positional encoding of event sequences.

        Args:
        ----
            time (torch.Tensor): true event times

        Returns:
        -------
            torch.Tensor with encoded times tensor
        """
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result

    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        kernel_size: int,
        dilation: int,
        include_zero_lag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size.

        Args:
        ----
            times (torch.Tensor): all times of shape = (bs, max_len)
            features (torch.Tensor): input of shape = (bs, max_len, in_channels)
            non_pad_mask (torch.Tensor): non-padding timestamps of shape = (bs, max_len)
            kernel_size (int): covolution kernel size
            dilation (int): convolution dilation
            include_zero_lag (bool): indicates if we should use zero-lag timestamp

        Returns:
        -------
            * delta_times - torch.Tensor of shape = (bs, kernel_size, max_len) with delta times value between current time and kernel_size true times before it
            * pre_conv_features - torch.Tensor of shape = (bs, kernel_size, max_len, in_channels) with corresponding input features of timestamps in delta_times
            * dt_mask - torch.Tensor of shape = (bs, kernel_size, max_len), bool tensor that indicates delta_times true values
        """
        # parameters
        padding = (
            (kernel_size - 1) * dilation if include_zero_lag else kernel_size * dilation
        )
        kernel = torch.eye(kernel_size).unsqueeze(1).to(times.device)
        in_channels = features.shape[2]

        # convolutions
        pre_conv_times = F.conv1d(
            times.unsqueeze(1), kernel, padding=padding, dilation=dilation
        )
        pre_conv_features = F.conv1d(
            features.transpose(1, 2),
            kernel.repeat(in_channels, 1, 1),
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        dt_mask = (
            F.conv1d(
                non_pad_mask.float().unsqueeze(1),
                kernel.float(),
                padding=padding,
                dilation=dilation,
            )
            .long()
            .bool()
        )

        # deleting extra values
        pre_conv_times = pre_conv_times[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ]
        pre_conv_features = pre_conv_features[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ]
        dt_mask = dt_mask[
            :, :, : -(padding + dilation * (1 - int(include_zero_lag)))
        ] * non_pad_mask.unsqueeze(1)

        # updating shape
        bs, L, dim = features.shape
        pre_conv_features = pre_conv_features.reshape(bs, dim, kernel_size, L)

        # computing delte_time and deleting masked values
        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0
        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))
        pre_conv_features[~dt_mask, :] = 0

        return delta_times, pre_conv_features, dt_mask

    def forward(
        self, times: torch.Tensor, features: torch.Tensor, non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Neural net layer forward pass.

        Args:
        ----
            times (torch.Tensor): event times of shape = (bs, L)
            features (torch.Tensor): event features of shape = (bs, L, in_channels)
            non_pad_mask (torch.Tensor): mask that indicates non pad values shape = (bs, L)

        Returns:
        -------
            torch.Tensor of shape = (bs, L, out_channels)
        """
        delta_times, features_kern, dt_mask = self.__conv_matrix_constructor(
            times,
            features,
            non_pad_mask,
            self.kernel_size,
            self.dilation,
            self.include_zero_lag,
        )
        bs, k, L = delta_times.shape

        kernel_values = self.kernel(self.__temporal_enc(delta_times))
        kernel_values[~dt_mask, ...] = 0

        out = features_kern.unsqueeze(-1) * kernel_values
        out = out.sum(dim=(1, 3))

        out = out + self.skip_connection(features.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class ContConv1dSim(nn.Module):
    """Continuous convolution layer for a sequence with auxiliary (simulated) random timestamps for intensity function calculation."""

    def __init__(
        self, kernel: nn.Module, kernel_size: int, in_channels: int, out_channels: int
    ):
        """Initialize Continuous convolutional layer for a sequences with auxiliary simulated times.

        Args:
        ----
            kernel (torch.nn.Module): Kernel neural net that takes (*,1) as input and returns (*, in_channles, out_channels) as output
            kernel_size (int): convolution layer kernel size
            in_channels (int): features input size
            out_channels (int): output size
        """
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / self.in_channels)
                for i in range(self.in_channels)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def __temporal_enc(self, time: torch.Tensor) -> torch.Tensor:
        """Positional encoding of event sequences.

        Args:
        ----
            time (torch.Tensor): true event times

        Returns:
        -------
            torch.Tensor with encoded times tensor
        """
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result

    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        true_times: torch.Tensor,
        true_features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        kernel_size: int,
        sim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size.

        Args:
        ----
            times (torch.Tensor): all times of shape = (bs, max_len)
            true_times (torch.Tensor): all times of shape = (bs, max_len)
            true_features (torch.Tensor): input of shape = (bs, max_len, in_channels)
            non_pad_mask (torch.Tensor): non-padding timestamps of shape = (bs, max_len)
            kernel_size (int): covolution kernel size
            sim_size (int): where to consider similarity.

        Returns:
        -------
            * delta_times - torch.Tensor of shape = (bs, kernel_size, max_len) with delta times value between current time and kernel_size true times before it
            * pre_conv_features - torch.Tensor of shape = (bs, kernel_size, max_len, in_channels) with corresponding input features of timestamps in delta_times
            * dt_mask - torch.Tensor of shape = (bs, kernel_size, max_len), bool tensor that indicates delta_times true values
        """
        # parameters
        padding = (kernel_size) * 1
        kernel = torch.eye(kernel_size).unsqueeze(1).to(times.device)
        in_channels = true_features.shape[2]

        # true values convolutions
        pre_conv_times = F.conv1d(
            true_times.unsqueeze(1), kernel, padding=padding, dilation=1
        )
        pre_conv_features = F.conv1d(
            true_features.transpose(1, 2),
            kernel.repeat(in_channels, 1, 1),
            padding=padding,
            dilation=1,
            groups=true_features.shape[2],
        )
        dt_mask = (
            F.conv1d(
                non_pad_mask.float().unsqueeze(1),
                kernel.float(),
                padding=padding,
                dilation=1,
            )
            .long()
            .bool()
        )

        # deleting extra values
        if padding > 0:
            pre_conv_times = pre_conv_times[:, :, : -(padding + 1)]
            pre_conv_features = pre_conv_features[:, :, : -(padding + 1)]
            dt_mask = dt_mask[:, :, : -(padding + 1)] * non_pad_mask.unsqueeze(1)
        else:
            dt_mask = dt_mask * non_pad_mask.unsqueeze(1)

        # reshaping features output
        bs, L, dim = true_features.shape
        pre_conv_features = pre_conv_features.reshape(bs, dim, kernel_size, L)

        # adding sim_times
        pre_conv_times = pre_conv_times.unsqueeze(-1).repeat(1, 1, 1, sim_size + 1)
        pre_conv_times = pre_conv_times.flatten(2)
        if sim_size > 0:
            pre_conv_times = pre_conv_times[..., :-sim_size]

        pre_conv_features = pre_conv_features.unsqueeze(-1).repeat(
            1, 1, 1, 1, sim_size + 1
        )
        pre_conv_features = pre_conv_features.flatten(3)
        if sim_size > 0:
            pre_conv_features = pre_conv_features[..., :-sim_size]

        dt_mask = dt_mask.unsqueeze(-1).repeat(1, 1, 1, sim_size + 1)
        dt_mask = dt_mask.flatten(2)
        dt_mask = dt_mask[..., sim_size:]

        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0

        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))
        pre_conv_features[~dt_mask, :] = 0
        return delta_times, pre_conv_features, dt_mask

    def forward(
        self,
        times: torch.Tensor,
        true_times: torch.Tensor,
        true_features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        sim_size: int,
    ) -> torch.Tensor:
        """Neural net layer forward pass.

        Args:
        ----
            times (torch.Tensor): all times (prepended with zeros by .__ad_bos) of shape = (bs, (sim_size+1)*(max_len-1)+1)
            true_times (torch.Tensor): true times of shape = (bs, max_len)
            true_features (torch.Tensor): input (aka 'encoded_outpup') of shape = (bs, max_len, in_channels)
            non_pad_mask (torch.Tensor): non-padding timestamps of shape = (bs, max_len)
            sim_size (int): simulated times size

        Returns:
        -------
            torch.Tensor of shape = (bs, L, out_channels)
        """
        delta_times, features_kern, dt_mask = self.__conv_matrix_constructor(
            times, true_times, true_features, non_pad_mask, self.kernel_size, sim_size
        )

        bs, k, L = delta_times.shape
        kernel_values = self.kernel(self.__temporal_enc(delta_times))
        kernel_values[~dt_mask, ...] = 0
        out = features_kern.unsqueeze(-1) * kernel_values
        out = out.sum(dim=(1, 3))

        return out
