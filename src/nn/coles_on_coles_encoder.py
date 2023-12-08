"""The file with the ColesOnColes encoder model."""
from typing import Optional

import numpy as np
import torch
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.coles.split_strategy import AbsSplit, SampleSlices
from ptls.nn import RnnEncoder, TrxEncoder
from torch import nn

from .pretrained_seq_encoder import PretrainedRnnSeqEncoder


def is_seq_feature(k: str, x):
    """Check is value sequential feature.

    Parameters
    ----------
    k:
        feature_name
    x:
        value for check

    """
    if k == "event_time":
        return True
    if k.startswith("target"):
        return False
    if type(x) is np.ndarray:
        return False
    if type(x) is torch.Tensor and len(x.shape) == 1:
        return False
    return True


class CoLESonCoLESEncoder(nn.Module):
    """Coles on coles embeddings model."""

    def __init__(
        self,
        trx_encoder: TrxEncoder,
        frozen_enc_type: str,
        frozen_enc_weight_path: str,
        learning_enc_type: str,
        intermediate_size: int,
        hidden_size: int,
        col_time: str = "event_time",
        encoding_seq_len: int = 20,
        encoding_step: int = 1,
        training_splitter: Optional[AbsSplit] = None,
        is_reduce_sequence: bool = True,
    ) -> None:
        """Model for training CoLES on CoLES embeddings obtained via seq2vec strategy.

        Args:
        ----
            trx_encoder:
                TrxEncoder of the first encoder.
            frozen_enc_type:
                Type of the first encoder.
            frozen_enc_weight_path:
                Path to the state dict of first encoder.
            learning_enc_type:
                Type of SeqEncoder which is to be trained on embeddings
            intermediate_size:
                Output size of first, frozen encoder, input size of second.
            hidden_size:
                Output size of second encoder and the overall output size.
            col_time:
                name of the column with events datettime
            encoding_seq_len:
                sequence length which is used to embed raw data
                (to obtain 1 embedding, we use encoding_seq_len timestamps)
            training_splitter:
                splitter used to train CoLES
            training_mode:
                whether a model is training or not
            encoding_step: 
                the step with which to encode sequences.
            is_reduce_sequence: 
                whether to reduce the encoded sequence to a single embedding.
        """
        super().__init__()
        self.frozen_encoder = PretrainedRnnSeqEncoder(
            path_to_dict=frozen_enc_weight_path,
            trx_encoder=trx_encoder,
            hidden_size=intermediate_size,
            type=frozen_enc_type,
            is_reduce_sequence=True,
        )

        self.learning_encoder = RnnEncoder(
            input_size=intermediate_size,
            hidden_size=hidden_size,
            type=learning_enc_type,
            is_reduce_sequence=is_reduce_sequence,
        )

        self.learning_encoder.is_reduce_sequence = is_reduce_sequence

        self.training_splitter = training_splitter or SampleSlices(
            split_count=5, cnt_max=150, cnt_min=15
        )

        self.col_time = col_time

        self.encoding_seq_len = encoding_seq_len
        self.encoding_step = encoding_step

    def _encode(self, x: PaddedBatch):
        dates = x.payload[self.col_time]  # (B, T)
        dates_len = dates.shape[1]
        start_pos = torch.arange(
            0, dates_len - self.encoding_seq_len, self.encoding_step, device="cuda"
        )

        def encode():
            for s in start_pos:
                payload = {
                    k: v[:, s : s + self.encoding_seq_len]
                    for k, v in x.payload.items()
                    if is_seq_feature(k, v)
                }
                seq_lens = torch.minimum(
                    torch.maximum(x.seq_lens - s, torch.zeros_like(x.seq_lens)),
                    torch.full_like(x.seq_lens, self.encoding_seq_len),
                )
                pb = PaddedBatch(payload, seq_lens) # type: ignore
                yield self.frozen_encoder(pb).detach()

        emb_sequences = torch.stack(
            [*encode()], dim=1
        )  # (B, T - self.encoding_seq_len, C)

        seq_lens = (
            start_pos.expand(x.seq_lens.size(0), start_pos.size(0))
            < x.seq_lens.unsqueeze(-1)
        ).sum(dim=-1)
        return PaddedBatch(payload=emb_sequences, length=seq_lens) # type: ignore

    def forward(self, x: PaddedBatch):
        """Pass x through the encoder."""
        embeddings = self._encode(x)

        return self.learning_encoder(embeddings)
