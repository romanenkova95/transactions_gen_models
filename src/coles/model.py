"""CoLES model"""
from typing import Callable, Dict
from itertools import chain

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.coles import CoLESModule
from ptls.frames.coles.split_strategy import SampleSlices


class CustomCoLES(CoLESModule):
    """
    Custom coles module inhereted from ptls coles module.
    """

    def __init__(
        self,
        optimizer_partial: Callable,
        lr_scheduler_partial: Callable,
        sequence_encoder: SeqEncoderContainer,
    ) -> None:
        """Overrided initialize method, which is suitable for our tasks

        Args:
            optimizer_partial (Callable): Partial initialized torch optimizer (with parameters)
            lr_scheduler_partial (Callable): Partial initialized torch lr scheduler
                (with parameters)
            sequence_encoder (SeqEncoderContainer): Ptls sequence encoder
                (including sequence encoder and single transaction encoder)
        """
        super().__init__(
            seq_encoder=sequence_encoder,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
        )
        self.sequence_encoder_model = sequence_encoder

    def get_seq_encoder_weights(self) -> Dict:
        """Get weights of the sequnce encoder in torch format

        Returns:
            dict: Encoder weights
        """
        return self.sequence_encoder_model.state_dict()

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y


def is_seq_feature(k: str, x):
    """Check is value sequential feature
    Synchronized with ptls.data_load.feature_dict.FeatureDict.is_seq_feature

                    1-d        2-d
    event_time | True      True
    target_    | False     False  # from FeatureDict.is_seq_feature
    tensor     | False     True

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


class CoLESonCoLES(CoLESModule):
    """
    Coles on coles embeddings model
    """

    def __init__(
        self,
        frozen_encoder: SeqEncoderContainer,
        learning_encoder: AbsSeqEncoder,
        optimizer_partial: Callable,
        lr_scheduler_partial: Callable,
        col_time: str = "event_time",
        encoding_seq_len: int = 20,
        training_splitter: SampleSlices = SampleSlices(
            split_count=5, cnt_max=150, cnt_min=15
        ),
        training_mode: bool = True,
    ) -> None:
        """
        Model for training CoLES on CoLES embeddings obtained via seq2vec strategy.
        Args:
            frozen_encoder: pretrained CoLES model (used to embed raw data)
            learning_encoder: SeqEncoder which is to be trained on embeddings
            optimizer_partial: partially initialized torch optimizer (with parameters)
            lr_scheduler_partial: partially initialized lr scheduler (with parameters)
            col_time: name of the column with events datettime
            encoding_seq_len: sequence length which is used to embed raw data
                (to obtain 1 embedding, we use encoding_seq_len timestamps)
            training_splitter: splitter used to train CoLES
            training_mode: whether a model is training or not
        """
        super().__init__(
            seq_encoder=frozen_encoder,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
        )

        self.learning_encoder = learning_encoder

        self.encoding_seq_len = encoding_seq_len
        self.training_splitter = training_splitter

        self.col_time = col_time

        self.training_mode = training_mode

    def _encode(self, x: PaddedBatch):
        dates = x.payload[self.col_time]  # (B, T)
        dates_len = dates.shape[1]
        start_pos = np.arange(0, dates_len - self.encoding_seq_len, 1)

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
                yield self._seq_encoder(pb).detach().cpu()

        emb_sequences = torch.stack(
            [*encode()], dim=1
        )  # (B, T - self.encoding_seq_len, C)
        emb_sequences = torch.cat(
            [emb_sequences[:, : self.encoding_seq_len], emb_sequences], dim=1
        )  # (B, T, C)
        return emb_sequences.to(x.device)

    def _split(self, x: torch.Tensor, dates: torch.Tensor):
        def split():
            for elem in x:
                indexes = self.training_splitter.split(dates)
                yield [elem[ix] for ix in indexes]

        emb_sequences = list(chain(*split()))  # B
        emb_sequences = pad_sequence(emb_sequences, batch_first=True)
        seq_lens = (emb_sequences != 0).all(dim=-1).sum(dim=-1)

        return PaddedBatch(payload=emb_sequences, length=seq_lens)

    def forward(self, x):
        dates = x.payload[self.col_time]
        x = self._encode(x)
        if not self.training_mode:
            x = PaddedBatch(payload=x, length=(x != 0).all(dim=-1).sum(dim=-1))
        else:
            x = self._split(x, dates)
        return self.learning_encoder(x)

    def shared_step(self, x, y):
        y = list(
            chain(*[[elem.item()] * self.training_splitter.split_count for elem in y])
        )
        y = torch.tensor(y).to(x.device)
        y_h = self._head(self(x))
        return y_h, y
