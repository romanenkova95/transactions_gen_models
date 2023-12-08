"""The file with the TS2Vec encoder."""
from typing import Optional

import torch
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from torch import nn

from .ts2vec_components import DilatedConvEncoder


class ConvEncoder(AbsSeqEncoder):
    """Convolutional sequence encoder for the TS2Vec model."""

    def __init__(
        self,
        kernel_size: int,
        hidden_size: int,
        input_size: Optional[int] = None,
        num_layers: int = 10,
        dropout: float = 0,
        is_reduce_sequence: bool = False,
        reducer: str = "maxpool",
    ) -> None:
        """Initialize ConvEncoder.

        Args:
        ----
            kernel_size (int): kernel size
            hidden_size (int): hidden size (aka embedding dim for this backbone)
            input_size (int or None): input size (if None, use output size of TrxEncoder)
            num_layers (int): number of layers (convolutional blocks)
            dropout (float): dropout probability
            is_reduce_sequence (bool): if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            reducer (str): type of reducer
        """
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size

        self.feature_extractor = DilatedConvEncoder(
            input_size, # type: ignore
            [input_size] * num_layers + [hidden_size], # type: ignore
            kernel_size=kernel_size,
        )

        self.repr_dropout = nn.Dropout(dropout)

        self.reducer = reducer

    def forward(self, x: PaddedBatch) -> torch.Tensor:
        """Encode input batch of sequences.

        Args:
        ----
            x (PaddedBatch): batch of input sequences (ptls format)

        Returns:
        -------
            output of encoder
        """
        # conv encoder
        input_ = x.payload.transpose(1, 2)  # type: ignore # B x Ch x T

        out = self.repr_dropout(self.feature_extractor(input_))  # B x Co x T
        out = out.transpose(1, 2)  # B x T x Co

        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            out = out.payload.max(dim=1).values # type: ignore

        return out  # type: ignore # x: B x T x input_dims


class ConvSeqEncoder(SeqEncoderContainer):
    """Pytorch-lifestream container wrapper for convoluitonal sequence encoder."""

    def __init__(
        self,
        trx_encoder: Optional[TrxEncoder] = None,
        input_size: Optional[int] = None,
        is_reduce_sequence: bool = False,
        **seq_encoder_params,
    ) -> None:
        """Initialize ConvSeqEncoder.

        Args:
        ----
            trx_encoder (TrxEncoder or None): transactions encoder
            input_size (int or None): input size (output size of feature embeddings)
            is_reduce_sequence (bool): if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            **seq_encoder_params: other sequence encoder parameters
        """
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=ConvEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )
