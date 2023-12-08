from typing import Optional

import torch
from torch import nn

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn import TrxEncoder
from ptls.data_load.padded_batch import PaddedBatch

from .cotic_components import CCNN


class CoticEncoder(AbsSeqEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_types: int,
        kernel: nn.Module,
        num_layers: int = 10,
        kernel_size: int = 5,
        is_reduce_sequence: Optional[bool] = False,
        reducer: str = "maxpool",
    ) -> None:
        """Continous convoluitonal sequence encoder for COTIC model.

        Args:
        ----
            input_size (int) - input size for CCNN (output size of feature embeddings)
            hidden_size (int) - size of the output embeddings of the encoder
            num_types (int) - number of event types in in the dataset
            kernel (nn.Module) - kernel (MLP, by default)
            num_layers (int) - number of continuous convolutional layers
            kernel_size (int) - kernel size
            is_reduce_sequence (bool) - if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            reducer (str) - type of reducer (only 'maxpool' is available now)
        """
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size

        self.kernel = kernel

        self.feature_extractor = CCNN(
            in_channels=input_size,
            kernel_size=kernel_size,
            nb_filters=hidden_size,
            nb_layers=num_layers,
            num_types=num_types,
            kernel=self.kernel,
        )

        self.reducer = reducer

    def forward(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            times (torch.Tensor) - timestamps (extracted from PaddedBatch)
            features (torch.Tensor) - event type features (extracted from PaddedBatch) and passed through nn.Embedding

        Returns:
        -------
            torch.Tensor with model output
        """
        out = self.feature_extractor(event_times, event_types)  # B x Co x T

        if self.is_reduce_sequence:
            if self.reducer == "maxpool":
                out = out.max(dim=1).values
            else:
                raise NotImplementedError("Unknown reducer.")

        return out


class CoticSeqEncoder(SeqEncoderContainer):
    def __init__(
        self,
        input_size: int,
        trx_encoder: Optional[TrxEncoder] = None,
        is_reduce_sequence: bool = False,
        col_time: str = "event_time",
        col_type: str = "mcc_code",
        **seq_encoder_params,
    ) -> None:
        """Pytorch-lifestream container wrapper for Continous convoluitonal sequence encoder.

        Args:
        ----
            trx_encoder (TrxEncoder=None) - we do not use TrxEncoder in this model as we need to keep initial times and features
            input_size (int) - input size for CCNN (output size of feature embeddings)
            is_reduce_sequence (bool) - if True, use reducer and work in the 'seq2vec' mode, else work in 'seq2seq'
            col_time (str) - name of the field (in PaddedBatch.payload) containig event times
            col_type (str) - name of the field (in PaddedBatch.payload) containig event types
            **seq_encoder_params - other sequence encoder parameters
        """
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=CoticEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )
        self.col_time = col_time
        self.col_type = col_type

    def _extract_times_and_features(self, x: PaddedBatch) -> tuple[torch.Tensor]:
        """Extract event times and types from the input in ptls format.

        Args:
        ----
            x (PaddedBatch) - input batch of data

        Returns a tuple of:
            * torch.Tensor containing event times
            * torch.Tensor containing event types
        """
        event_times = x.payload[self.col_time].float()
        event_types = x.payload[self.col_type]

        return event_times, event_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            x (PaddedBatch) - input batch from CoticDataset (i.e. ColesDataset with NoSplit())

        Returns:
        -------
            torch.Tensor with model output
        """
        event_times, event_types = self._extract_times_and_features(x)

        out = self.seq_encoder(event_times, event_types)
        return out
