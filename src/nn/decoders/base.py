"""File with the base decoder class."""

from typing import Optional

from torch import nn


class AbsDecoder(nn.Module):
    """Abstract decoder class. Only requirement is to specify the output_size.

    Properties:
        output_size (int): size of the channel dimension of the decoder output.
    """

    def __init__(self, output_size: Optional[int] = None) -> None:
        """Initialize internal module state, set output_size if given.

        Args:
        ----
            output_size (Optional[int], optional):
                you could set the output_size here, or by overloading the property.
                Defaults to None, in which case one should overload.
        """
        super().__init__()
        if output_size:
            self._output_size = output_size

    @property
    def output_size(self):
        """The size of output embeddings."""
        if hasattr(self, "_output_size"):
            return self._output_size
        else:
            raise NotImplementedError(
                "Output size wasn't provided on init, and wasn't implemented in child"
            )

    def forward(self, x, *args, **kwargs):
        """Pass x through the decoder, without changing if not overloaded."""
        return x
