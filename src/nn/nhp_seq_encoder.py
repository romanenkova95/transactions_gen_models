from typing import Optional, Tuple

import torch
import torch.nn as nn 

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from .nhp_components import ContTimeLSTMCell, restruct_batch


class NHPEncoder(AbsSeqEncoder):
    """Continuous-time LSTM sequence encoder for the NHP model."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_types: int,
        beta: float,
        bias: bool,
        max_steps: Optional[int],
        max_decay_time: float,
        num_event_types_pad: int,
        pad_token_id: int,
        is_reduce_sequence: Optional[bool] = False,
        reducer: str = "maxpool",
    ) -> None:
        """Initialize continous-time LSTM sequence encoder for the NHP model.

        Args:
        ----
            input_size (int): input size for CCNN (output size of feature embeddings)
            hidden_size (int): size of the output embeddings of the encoder
            num_types (int): number of event types in in the dataset
            beta (float): beta in nn.Softplus for ContTimeLSTMCell
            bias (bool): if True, include bias in the intensity layer
            max_steps (int): if not None, crop all the sequences up to 'max_steps'
            max_decay_time (float): restrict maximum dt for the decay of the LSTM hidden state
            num_event_types_pad (int): total number of events, including padding type 
            pad_token_id (int): event type used for padding (num_event_types in EasyTPP pipeline)
            is_reduce_sequence (bool): if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            reducer (str): type of reducer (only 'maxpool' is available now)
        """
        super().__init__(is_reduce_sequence=is_reduce_sequence) 

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_types = num_types
        self.beta = beta
        self.bias = bias

        self.max_steps = max_steps
        self.max_decay_time = max_decay_time

        self.pad_token_id = pad_token_id

        self.rnn_cell = ContTimeLSTMCell(
            embed_dim=input_size,
            hidden_dim=hidden_size,
            num_event_types_pad=num_event_types_pad,
            pad_token_id=pad_token_id,
            beta=beta,
        )

        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_types, self.bias),
            nn.Softplus(self.beta),
        )

        self.reducer = reducer

    def init_state(self, batch_size: int, device: str) -> Tuple[torch.Tensor]:
        """Initialize hidden and cell states of the encoder.

        Args:
        ----
            batch_size (int): size of batch data
            device (str): 'cpu' or 'cuda' if available

        Returns:
        -------
            * h_t - torch.Tensor of LSTM hidden states
            * c_t - torch.Tensor of LSTM cell states
            * c_bar - torch.Tensor of LSTM cell bar states 
        """
        h_t, c_t, c_bar = torch.zeros(
            batch_size, 3 * self.hidden_size, device=device
        ).chunk(3, dim=1)

        return h_t, c_t, c_bar

    def run_batch(self, time_delta_seq: torch.Tensor, event_seq: torch.Tensor) -> Tuple[torch.Tensor]:
        """Pass batch through the model, return hidden states and decayed states for log-likelihood loss computation.

        Args:
            inputs (Tuple of torch.Tensors): 
                * time_delta_seq - time deltas between events
                * event_seq - event types

        Returns:
            * hiddens_stack (torch.Tensor): hidden states of contimuous-time LSTM
            * decay_states_stack (torch.Tensor): hidden states of contimuous-time LSTM after exponential decay 
        """
        all_hiddens = []
        all_outputs = []
        all_cells = []
        all_cell_bars = []
        all_decays = []

        # last event has no time label
        max_seq_length = (
            self.max_steps if self.max_steps is not None else event_seq.size(1) - 1
        )

        batch_size = len(event_seq)
        h_t, c_t, c_bar_i = self.init_state(batch_size, time_delta_seq.device)

        # if only one event, then we dont decay
        if max_seq_length == 1:
            types_sub_batch = event_seq[:, 0]
            # x_t = self.layer_type_emb(types_sub_batch)
            cell_i, c_bar_i, decay_i, output_i = self.rnn_cell(
                types_sub_batch, h_t, c_t, c_bar_i
            )

            # Append all output
            all_outputs.append(output_i)
            all_decays.append(decay_i)
            all_cells.append(cell_i)
            all_cell_bars.append(c_bar_i)
            all_hiddens.append(h_t)
        else:
            # Loop over all events
            for i in range(max_seq_length):
                if i == event_seq.size(1) - 1:
                    dt = torch.ones_like(time_delta_seq[:, i]) * self.max_decay_time
                else:
                    dt = time_delta_seq[:, i + 1]
                types_sub_batch = event_seq[:, i]

                # cell_i  (batch_size, process_dim)
                cell_i, c_bar_i, decay_i, output_i = self.rnn_cell(
                    types_sub_batch, h_t, c_t, c_bar_i
                )

                # States decay - Equation (7) in the paper
                c_t, h_t = self.rnn_cell.decay(
                    cell_i, c_bar_i, decay_i, output_i, dt[:, None]
                )

                # Append all output
                all_outputs.append(output_i)
                all_decays.append(decay_i)
                all_cells.append(cell_i)
                all_cell_bars.append(c_bar_i)
                all_hiddens.append(h_t)

        # (batch_size, max_seq_length, hidden_dim)
        cell_stack = torch.stack(all_cells, dim=1)
        cell_bar_stack = torch.stack(all_cell_bars, dim=1)
        decay_stack = torch.stack(all_decays, dim=1)
        output_stack = torch.stack(all_outputs, dim=1)

        # [batch_size, max_seq_length, hidden_dim]
        hiddens_stack = torch.stack(all_hiddens, dim=1)

        # [batch_size, max_seq_length, 4, hidden_dim]
        decay_states_stack = torch.stack(
            (cell_stack, cell_bar_stack, decay_stack, output_stack), dim=2
        )

        return hiddens_stack, decay_states_stack
    
    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            inputs (Tuple[torch.Tensor]): inputs as passed to .run_batch() method above

        Returns: 
        -------
            torch.Tensor with model output
        """
        time_delta_seq, event_seq = inputs

        out = self.run_batch(time_delta_seq, event_seq)[0] # take only hidden states for embeddings
        
        if self.is_reduce_sequence:
            if self.reducer == "maxpool":
                out = out.max(dim=1).values
            else:
                raise NotImplementedError("Unknown reducer.")
        return out


class NHPSeqEncoder(SeqEncoderContainer):
    """Pytorch-lifestream container wrapper for NHP sequence encoder."""
    def __init__(
        self,
        input_size: int,
        trx_encoder: Optional[TrxEncoder] = None,
        is_reduce_sequence: bool = False,
        col_time: str = "event_time",
        col_type: str = "mcc_code",
        **seq_encoder_params,
    ) -> None:
        """Initialize pytorch-lifestream container wrapper for NHP sequence encoder.

        Args:
        ----
            input_size (int): input size for CCNN (output size of feature embeddings)
            trx_encoder (TrxEncoder=None): we do not use TrxEncoder in this model as we need to keep initial times and features
            is_reduce_sequence (bool): if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            col_time (str): name of the field (in PaddedBatch.payload) containig event times
            col_type (str): name of the field (in PaddedBatch.payload) containig event types
            **seq_encoder_params: other sequence encoder parameters
        """
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=NHPEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

        self.col_time = col_time
        self.col_type = col_type

    def forward(self, x: PaddedBatch) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            x (PaddedBatch): input batch from CoticDataset (i.e. ColesDataset with NoSplit())

        Returns:
        -------
            torch.Tensor with model output
        """
        _, time_delta, event_types, _, _, _ = restruct_batch(
            x,
            col_time=self.col_time, 
            col_type=self.col_type, 
            pad_token_id=self.seq_encoder.pad_token_id, 
            num_types=self.seq_encoder.num_types
        )

        inputs = (time_delta, event_types)
        out = self.seq_encoder(inputs)

        return out