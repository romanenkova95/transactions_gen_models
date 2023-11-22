from typing import Optional

import numpy as np
import torch

from ptls.nn import RnnSeqEncoder, RnnEncoder
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.coles.split_strategy import AbsSplit, SampleSlices

import torch.nn as nn


def is_seq_feature(k: str, x):
    """Check is value sequential feature
    Parameters
    ----------
    k:
        feature_name
    x:
        value for check

    Returns
    -------

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


class PretrainedRnnSeqEncoder(RnnSeqEncoder):
    """Pretrained network layer which makes representation for single transactions

    Args:
        path_to_dict (str): Path to state dict of a pretrained RnnSeqEncoder
        **seq_encoder_params: Params for RnnSeqEncoder initialization
    """

    def __init__(self, path_to_dict: Optional[str] = None, **seq_encoder_params):
        super().__init__(**seq_encoder_params)

        if path_to_dict is not None:
            self.load_state_dict(torch.load(path_to_dict))

        # freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    @property
    def output_size(self) -> int:
        """Returns embedding size of a single transaction"""
        return self.embedding_size


class CoLESonCoLESEncoder(nn.Module):
    """
    Coles on coles embeddings model
    """

    def __init__(
        self,
        frozen_encoder: PretrainedRnnSeqEncoder,
        learning_encoder: RnnEncoder,
        col_time: str = "event_time",
        encoding_seq_len: int = 20,
        encoding_step: int = 1,
        training_splitter: Optional[AbsSplit] = None,
        is_reduce_sequence: bool = True,
    ) -> None:
        """
        Model for training CoLES on CoLES embeddings obtained via seq2vec strategy.
        Args:
            frozen_encoder: pretrained CoLES model (used to embed raw data)
            learning_encoder: SeqEncoder which is to be trained on embeddings
            col_time: name of the column with events datettime
            encoding_seq_len: sequence length which is used to embed raw data
                (to obtain 1 embedding, we use encoding_seq_len timestamps)
            training_splitter: splitter used to train CoLES
            training_mode: whether a model is training or not
        """
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.learning_encoder = learning_encoder

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
            0, dates_len - self.encoding_seq_len, self.encoding_step
        )

        def encode():
            for s in start_pos:
                torch.cuda.empty_cache()
                payload = {
                    k: v[:, s : s + self.encoding_seq_len]
                    for k, v in x.payload.items()
                    if is_seq_feature(k, v)
                }
                seq_lens = torch.minimum(
                    torch.maximum(x.seq_lens - s, torch.zeros_like(x.seq_lens)),
                    torch.full_like(x.seq_lens, self.encoding_seq_len),
                )
                pb = PaddedBatch(payload, seq_lens)
                yield self.frozen_encoder(pb).detach().cpu()

        emb_sequences = torch.stack(
            [*encode()], dim=1
        )  # (B, T - self.encoding_seq_len, C)

        seq_lens = x.seq_lens.cpu()
        seq_lens = (
            start_pos.expand(seq_lens.size(0), start_pos.size(0))
            < seq_lens.unsqueeze(-1)
        ).sum(dim=-1)
        return PaddedBatch(payload=emb_sequences.to(x.device), length=seq_lens)

    def forward(self, x: PaddedBatch):
        embeddings = self._encode(x)

        return self.learning_encoder(embeddings)
