from typing import Tuple, Optional
import torch
from torch import nn, Tensor
from src.generation.decoders.base import AbsDecoder
from src.generation.decoders import LSTMDecoder


class LSTMCellDecoder(AbsDecoder):
    """An NLP-style LSTM-based decoder. 
    Restores a sequence of embeddings from a single embedding

    Attributes:
        cell (nn.LSTMCell): 
            The lstm cell, used by this module.
        projector (nn.Sequential(nn.Linear, nn.Relu)): 
            The linear layer, used to reshape lstmcell's outputted hidden state to match input_size.
        lstm (nn.LSTM):
            The LSTM, used for any layers other than the first one
        hidden_size (int):
            The hidden size of the LSTMCell
        output_size (int):
            Size of the channel dimension of the lstmcell output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        proj_size: int = 0
    ) -> None:
        """Initializes LSTMCellDecoder's internal state.

        Args:
            input_size (int): 
                Input size of the lstmcell.
            hidden_size (int): 
                Hidden size of the lstmcell.
            num_layers (int, optional): 
                Number of lstm layers. Defaults to 1. If >1, adds num_layers-1 nn.LSTM layers after nn.LSTMCell.
            proj_size (int, optional): 
            Relevant if num_layers > 1. Sets the proj_size of appended nn.LSTM layers. Defaults to 0.
        """
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        self.lstm = LSTMDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers - 1
        ) if num_layers > 1 else nn.Identity()

        self.hidden_size = hidden_size

        if isinstance(self.lstm, AbsDecoder):
            self.output_size = self.lstm.output_size
        else:
            self.output_size = hidden_size

    def forward(self, input: Tensor, L: int, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """Runs the forward pass.

        Args:
            input (Tensor): 
                Input embedding, of size (batch_size, input_size).
            L (int): 
                Length of desired sequence.
            hx (Optional[Tuple[Tensor, Tensor]], optional): 
                Optionally, LSTMCell hidden & cell states. Defaults to None.

        Returns:
            Tensor: Generated sequence, of shape (batch_size, L, output_size).
        """
        B = input.shape[0]
        H = self.hidden_size
        hidden_state, cell_state = hx or (input.new_zeros(B, H), input.new_zeros(B, H))
        outputs_list = []
        for _ in range(L):
            hidden_state, cell_state = self.cell(input, (hidden_state, cell_state))
            outputs_list.append(hidden_state)
            input = self.projector(hidden_state)

        outputs_list.reverse()
        lstmcell_outputs = torch.relu(torch.stack(outputs_list, dim=1))
        return self.lstm(lstmcell_outputs)[0]
