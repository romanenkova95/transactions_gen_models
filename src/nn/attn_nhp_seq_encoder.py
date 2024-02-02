import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from .nhp_components import EncoderLayer, MultiHeadAttention, restruct_batch


class AttnNHPEncoder(AbsSeqEncoder):
    """Transformer-style sequence encoder for the A-NHP model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_ln: bool,
        time_emb_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_types: int,
        num_event_types_pad: int,
        pad_token_id: int,
        is_reduce_sequence: bool = False,
        reducer: str = "maxpool",
    ) -> None:
        """Initialize transformer-style sequence encoder for the A-NHP model.

        Args:
        ----
            input_size (int): input size for CCNN (output size of feature embeddings)
            hidden_size (int): size of the output embeddings of the encoder
            use_ln (bool): if True, add layer normalization
            time_emb_size (int): embedding size for temporal encoding
            num_layers (int): number of layers in the Encoder
            num_heads (int): number of attention heads
            dropout (float): dropout probability
            num_types (int): number of event types in in the dataset
            num_event_types_pad (int): total number of events, including padding type
            pad_token_id (int): event type used for padding (num_event_types in EasyTPP pipeline)
            is_reduce_sequence (bool): if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            reducer (str): type of reducer (only 'maxpool' is available now)
        """
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.input_size = input_size
        self.model_size = hidden_size // 2
        self.hidden_size = hidden_size
        self.num_types = num_types

        self.num_event_types_pad = num_event_types_pad
        self.pad_token_id = pad_token_id

        self.use_ln = use_ln
        self.time_emb_size = time_emb_size

        self.div_term = torch.exp(
            torch.arange(0, self.time_emb_size, 2)
            * -(math.log(10000.0) / self.time_emb_size)
        ).reshape(1, 1, -1)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        heads = []
        for _ in range(self.num_heads):
            heads.append(
                nn.ModuleList(
                    [
                        EncoderLayer(
                            self.model_size + self.time_emb_size,
                            MultiHeadAttention(
                                1,
                                self.input_size + self.time_emb_size,
                                self.model_size,
                                self.dropout,
                                output_linear=False,
                            ),
                            use_residual=False,
                            dropout=self.dropout,
                        )
                        for _ in range(self.num_layers)
                    ]
                )
            )
        self.heads = nn.ModuleList(heads)

        if self.use_ln:
            self.norm = nn.LayerNorm(self.model_size)
        self.inten_linear = nn.Linear(self.model_size * self.num_heads, self.num_types)
        self.softplus = nn.Softplus()

        self.layer_type_emb = nn.Embedding(
            self.num_event_types_pad, self.input_size, padding_idx=pad_token_id
        )

        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)
        self.eps = torch.finfo(torch.float32).eps

        self.reducer = reducer

    def compute_temporal_embedding(self, time: torch.Tensor) -> torch.Tensor:
        """Compute the temporal embedding.

        Args:
        ----
            time (torch.Tensor): [batch_size, seq_len], input event times

        Returns:
        -------
            torch.Tensor: [batch_size, seq_len, emb_size], temporal embeddings
        """
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.time_emb_size).to(time)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)

        return pe

    def seq_encoding(
        self, time_seqs: torch.Tensor, event_seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the sequence.

        Args:
        ----
            time_seqs (torch.Tensor): time seqs input, [batch_size, seq_len]
            event_seqs (torch.Tensor): event type seqs input, [batch_size, seq_len]

        Returns:
            a tuple of torch.Tensors
                * event embedding (concatenation of time and type embeddings)
                * time embedding
                * type embedding
        """
        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)

        # [batch_size, seq_len, hidden_size]
        type_emb = torch.tanh(self.layer_type_emb(event_seqs.long()))

        # [batch_size, seq_len, hidden_size*2]
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb

    def make_layer_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create a tensor to do masking on layers.

        Args:
        ----
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
        ------
            tensor: aim to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, seq_len]
        layer_mask = (
            (torch.eye(attention_mask.size(1)) < 1)
            .unsqueeze(0)
            .expand_as(attention_mask)
        )
        return layer_mask

    def make_combined_att_mask(
        self, attention_mask: torch.Tensor, layer_mask: torch.Tensor
    ) -> torch.Tensor:
        """Combined attention mask and layer mask.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask (tensor): mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """
        # [batch_size, seq_len, seq_len * 2]
        combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = torch.cat(
            [attention_mask, torch.ones_like(layer_mask)], dim=-1
        )
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=1)
        return combined_mask

    def apply_layer(
        self,
        init_cur_layer: torch.Tensor,
        time_emb: torch.Tensor,
        sample_time_emb: torch.Tensor,
        event_emb: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update the structure sequentially.

        Args:
        ----
            init_cur_layer (tensor): [batch_size, seq_len, hidden_size]
            time_emb (tensor): [batch_size, seq_len, hidden_size]
            sample_time_emb (tensor): [batch_size, seq_len, hidden_size]
            event_emb (tensor): [batch_size, seq_len, hidden_size]
            combined_mask (tensor): [batch_size, seq_len, hidden_size]

        Returns:
        -------
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = event_emb.size(1)
        for head_i in range(self.num_heads):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            for layer_i in range(self.num_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size*2]
                layer_ = torch.cat([cur_layer_, sample_time_emb], dim=-1)
                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size*2]
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = self.heads[head_i][layer_i]
                # compute the output
                enc_output = enc_layer(_combined_input, combined_mask)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection

                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                # event emb
                event_emb = torch.cat([enc_output[:, :seq_len, :], time_emb], dim=-1)

                if self.use_ln:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def run_batch(
        self,
        time_seqs: torch.Tensor,
        event_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call the model.

        Args:
        ----
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            event_seqs (tensor): [batch_size, seq_len], sequences of event types.
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for event sequences.
            sample_times (tensor, optional): [batch_size, seq_len, num_samples]. Defaults to None.

        Returns:
        -------
            tensor: states at sampling times, [batch_size, seq_len, num_samples].
        """
        event_emb, time_emb, type_emb = self.seq_encoding(time_seqs, event_seqs)
        init_cur_layer = torch.zeros_like(type_emb)
        layer_mask = self.make_layer_mask(attention_mask).to(attention_mask.device)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)
        cur_layer_ = self.apply_layer(
            init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask
        )

        return cur_layer_

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            inputs (Tuple[torch.Tensor]): inputs as passed to .run_batch() method above

        Returns:
        -------
            torch.Tensor with model output
        """
        time_seqs, event_seqs, attention_mask, sample_times = inputs

        out = self.run_batch(time_seqs, event_seqs, attention_mask, sample_times)

        if self.is_reduce_sequence:
            if self.reducer == "maxpool":
                out = out.max(dim=1).values
            else:
                raise NotImplementedError("Unknown reducer.")
        return out


class AttnNHPSeqEncoder(SeqEncoderContainer):
    """Pytorch-lifestream container wrapper for the A-NHP sequence encoder."""

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
            seq_encoder_cls=AttnNHPEncoder,
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
        time_seqs, _, event_types, _, attention_mask, _ = restruct_batch(
            x,
            col_time=self.col_time,
            col_type=self.col_type,
            pad_token_id=self.seq_encoder.pad_token_id,
            num_types=self.seq_encoder.num_types,
        )

        inputs = (
            time_seqs,
            event_types,
            attention_mask,
            None,
        )  # sample_times=None Q: is it ok???

        out = self.seq_encoder(inputs)

        return out
