from torch import nn, Tensor
import typing as tp
from src.nn.decoders.base import AbsDecoder


class LSTMDecoder(AbsDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int = 0,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=proj_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.output_size = proj_size

    def forward(self, x: Tensor, hx: tp.Optional[tp.Tuple[Tensor, Tensor]] = None):
        return self.lstm(x, hx)[0]
