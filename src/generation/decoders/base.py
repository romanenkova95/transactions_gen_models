from typing import Optional
from torch import nn


class AbsDecoder(nn.Module):
    """Abstract decoder class. Only requirement is to specify the output_size
    Properties:
        output_size (int): size of the channel dimension of the decoder output.
    """
    def __init__(self, output_size: Optional[int] = None) -> None:
        super().__init__()
        if output_size:
            self._output_size = output_size

    @property
    def output_size(self):
        if hasattr(self, "_output_size"):
            return self._output_size
        else:
            raise NotImplementedError("Output size wasn't provided on init, and wasn't implemented in child")

    def forward(self, x, *args, **kwargs):
        return x
