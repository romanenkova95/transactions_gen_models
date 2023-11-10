import torch
import torch.nn as nn

from typing import Tuple


class PredictionHead(nn.Module):
    """2-head network for return-time and event-type prediction."""

    def __init__(self, in_channels: int, num_types: int) -> None:
        """Initialize head network.

        Args:
            in_channels (int) - number of input channels
            num_types (int) - number of event types in the dataset
        """
        super().__init__()
        self.return_time_prediction = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.event_type_prediction = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(), nn.Linear(128, num_types)
        )

    def forward(self, enc_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            encoded_output (torch.Tensor) - hidden state, output of the core continuous convolutional part of the model

        Returns a tuple of:
            * torch.Tensor with predicted return time
            * torch.Tensor with predicted event type
        """
        return self.return_time_prediction(enc_output), self.event_type_prediction(
            enc_output
        )
