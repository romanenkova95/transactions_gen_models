import numpy as np
import torch


class CNNClassifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        num_classes: int = 2,
        sequence_length: int = 300,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_layers = int(np.log2(in_channels)) - 1
        self.convolutions = torch.nn.Sequential(
            *[
                torch.nn.Conv1d(
                    2 ** (k + 1),
                    2**k,
                    kernel_size,
                    stride,
                    padding="same",
                    dilation=dilation,
                    bias=bias,
                )
                for k in range(self.num_layers, -1, -1)
            ]
        )

        self.activation = torch.nn.Softmax(dim=-1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(sequence_length, num_classes),
            self.activation,
        )

    def forward(self, input):
        inp = torch.transpose(input.payload, 1, 2)  # (N, L, C) -> (N, C, L)
        l = inp.shape[-1]
        inp = torch.nn.functional.pad(inp, pad=(0, self.sequence_length - l))

        output = self.convolutions(inp)
        output = output.view(output.shape[0], -1)

        return self.linear(output)
