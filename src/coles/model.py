"""CoLES model"""
from typing import Callable
from functools import partial

from omegaconf import DictConfig

import torch

from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.frames.coles import CoLESModule


class MyCoLES(CoLESModule):

    def __init__(
        self,
        optimizer_partial: Callable,
        lr_scheduler_partial: Callable,
        sequence_encoder: SeqEncoderContainer
    ) -> None:
        super().__init__(
            seq_encoder=sequence_encoder,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial
        )
