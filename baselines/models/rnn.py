"""File with the RNN classifier model."""
import torch


class RNNClassifier(torch.nn.Module):
    """A simple classification model based on a recurrent neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        num_classes: int = 1,
    ):
        """Initialize internal state.

        Args:
        ----
            input_size (int): number of channels in the input data
            hidden_size (int): hidden size of the GRU
            num_layers (int): number of layers in the GRU
            bias (bool): whether the GRU uses bias weights
            batch_first (bool): if True, then the input and output tensors are provided
                as (batch, seq, feature) instead of (seq, batch, feature)
            dropout (float):  if non-zero, introduces a Dropout layer on
                the outputs of each GRU layer except the last layer, with dropout
                probability equal to dropout
            bidirectional (bool): whether the GRU is bidirectional or not
            num_classes (int): number of classes in the classification task
        """
        super().__init__()

        self.backbone = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        activation = torch.nn.Softmax(dim=-1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d * num_layers * hidden_size, num_classes), activation
        )

    def forward(self, input):
        """Pass input through the model."""
        # output, (h_n, c_n) = self.backbone(input.payload)
        _, h_n = self.backbone(input.payload)
        batch_size = h_n.shape[-2]
        h_n = h_n.view(batch_size, -1)
        return self.linear(h_n)
