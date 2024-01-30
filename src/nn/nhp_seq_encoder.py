from typing import Optional

import torch
import torch.nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from .nhp_components import ContTimeLSTMCell


class NHPEncoder(AbsSeqEncoder):
    """The continuous convolutional sequence encoder for COTIC."""

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
        """Continous convoluitonal sequence encoder for NHP model.

        Args:
        ----
            input_size (int): input size for CCNN (output size of feature embeddings)
            hidden_size (int): size of the output embeddings of the encoder
            num_types (int): number of event types in in the dataset

            beta (float)
            bias (bool)
            max_steps (int)
            max_decay_time (float)
            num_event_types_pad (int)
            pad_token_id (int)

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

    def init_state(self, batch_size, device):
        """Initialize hidden and cell states.

        Args:
            batch_size (int): size of batch data.

        Returns:
            list: list of hidden states, cell states and cell bar states.
        """
        h_t, c_t, c_bar = torch.zeros(
            batch_size, 3 * self.hidden_size, device=device
        ).chunk(3, dim=1)

        return h_t, c_t, c_bar

    def forward(self, inputs) -> torch.Tensor:
        """Forward pass through the model.

        Args:

        Returns:
            torch.Tensor with model output
        """

        time_delta_seq, event_seq = inputs

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
                    dt = time_delta_seq[:, i + 1]  # need to carefully check here
                types_sub_batch = event_seq[:, i]
                # x_t = self.layer_type_emb(types_sub_batch)

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


class NHPSeqEncoder(SeqEncoderContainer):
    """Pytorch-lifestream container wrapper for NHP sequence encoder."""

    def __init__(
        self,
        input_size: int,  # aka embd size
        trx_encoder: Optional[TrxEncoder] = None,
        is_reduce_sequence: bool = False,
        col_time: str = "event_time",
        col_type: str = "mcc_code",
        **seq_encoder_params,
    ) -> None:
        """Pytorch-lifestream container wrapper for NHP sequence encoder.

        Args:
        ----
            trx_encoder (TrxEncoder=None): we do not use TrxEncoder in this model as we need to keep initial times and features
            input_size (int): input size for CCNN (output size of feature embeddings)
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

    @staticmethod
    def make_attn_mask_for_pad_sequence(pad_seqs, pad_token_id):
        """Make the attention masks for the sequence.

        Args:
            pad_seqs (tensor): list of sequences that have been padded with fixed length
            pad_token_id (int): optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad

        Returns:
            np.array: a bool matrix of the same size of input, denoting the masks of the
            sequence (True: non mask, False: mask)


        Example:
        ```python
        seqs = [[ 1,  6,  0,  7, 12, 12],
        [ 1,  0,  5,  1, 10,  9]]
        make_attn_mask_for_pad_sequence(seqs, pad_index=12)
        >>>
            batch_non_pad_mask
            ([[ True,  True,  True,  True, False, False],
            [ True,  True,  True,  True,  True,  True]])
            attention_mask
            [[[ True  True  True  True  True  True]
              [False  True  True  True  True  True]
              [False False  True  True  True  True]
              [False False False  True  True  True]
              [False False False False  True  True]
              [False False False False  True  True]]

             [[ True  True  True  True  True  True]
              [False  True  True  True  True  True]
              [False False  True  True  True  True]
              [False False False  True  True  True]
              [False False False False  True  True]
              [False False False False False  True]]]
        ```
        """
        seq_num, seq_len = pad_seqs.shape

        # [batch_size, seq_len]
        seq_pad_mask = pad_seqs == pad_token_id

        # [batch_size, seq_len, seq_len]
        attention_key_pad_mask = torch.tile(seq_pad_mask[:, None, :], (1, seq_len, 1))

        subsequent_mask = torch.tile(
            torch.triu(torch.ones((seq_len, seq_len), dtype=bool), diagonal=0)[
                None, :, :
            ],
            (seq_num, 1, 1),
        ).to(pad_seqs.device)

        attention_mask = subsequent_mask | attention_key_pad_mask

        return attention_mask

    @staticmethod
    def make_type_mask_for_pad_sequence(pad_seqs, num_event_types):
        """Make the type mask.

        Args:
            pad_seqs (tensor): a list of sequence events with equal length (i.e., padded sequence)

        Returns:
            np.array: a 3-dim matrix, where the last dim (one-hot vector) indicates the type of event
        """
        type_mask = torch.zeros([*pad_seqs.shape, num_event_types], dtype=torch.int32)

        for i in range(1, num_event_types):
            type_mask[:, :, i] = pad_seqs == i

        return type_mask.to(pad_seqs.device)

    def _restruct_batch(self, x: PaddedBatch):
        event_times = x.payload[self.col_time].float()
        event_types = x.payload[self.col_type]

        time_delta = event_times[:, 1:] - event_times[:, :-1]
        time_delta = torch.nn.functional.pad(time_delta, (1, 0))  # EasyTPP format

        non_pad_mask = event_times.ne(0)

        event_times[~non_pad_mask] = self.seq_encoder.pad_token_id
        event_types[~non_pad_mask] = self.seq_encoder.pad_token_id
        time_delta[~non_pad_mask] = self.seq_encoder.pad_token_id

        type_mask = self.make_type_mask_for_pad_sequence(
            event_types, num_event_types=self.seq_encoder.num_types
        )
        attention_mask = self.make_attn_mask_for_pad_sequence(
            event_types, pad_token_id=self.seq_encoder.pad_token_id
        )

        return (
            event_times,
            time_delta,
            event_types,
            non_pad_mask.type(torch.int32),
            attention_mask,
            type_mask,
        )  # no need for event_times and attention_mask here

    def forward(self, x: PaddedBatch) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            x (PaddedBatch): input batch from CoticDataset (i.e. ColesDataset with NoSplit())

        Returns:
        -------
            torch.Tensor with model output
        """
        _, time_delta, event_types, _, _, _ = self._restruct_batch(x)

        inputs = (time_delta, event_types)
        hiddens_stack, decay_states_stack = self.seq_encoder(inputs)

        return hiddens_stack, decay_states_stack

    def get_embeddings(self, x):
        out = self.forward(x)[0]
        if self.is_reduce_sequence:
            if self.seq_encoder.reducer == "maxpool":
                out = out.max(dim=1).values
            else:
                raise NotImplementedError("Unknown reducer.")
        return out
