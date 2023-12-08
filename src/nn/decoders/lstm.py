"""File with the LSTM Decoder."""
import typing as tp

from torch import Tensor, nn

from .base import AbsDecoder


class LSTMDecoder(AbsDecoder):
    """Simple decoder, based on a seq2seq lstm. Basically a wrapper of nn.LSTM.

    Attributes
    ----------
        lstm (nn.LSTM): the LSTM, used by this module.
        output_size (int): the LSTM's output_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int = 0,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        """Initialize the LSTMDecoder class. All arguments are passed to nn.LSTM as-is."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=proj_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    @property
    def output_size(self):
        """Size of output embeddings."""
        return (
            self.proj_size
            or self.hidden_size * (self.bidirectional + 1) * self.num_layers
        )

    def forward(self, x: Tensor, hx: tp.Optional[tuple[Tensor, Tensor]] = None):
        """Do the forward pass. Arguments same as in nn.LSTM.forward.

        Args:
        ----
            x (Tensor): Input to the LSTM, of shape (batch_size, L, input_size)
            hx (tp.Optional[tuple[Tensor, Tensor]], optional): tuple of (hidden_state, cell_state). Defaults to None.

        Returns:
        -------
            Tensor: LSTM output of shape (batch_size, L, output_size).
        """
        return self.lstm(x, hx)[0]
