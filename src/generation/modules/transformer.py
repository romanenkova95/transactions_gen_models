from typing import Any, Optional
from omegaconf import DictConfig
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.bert import MLMPretrainModule as ptlsMLMPretrainModule
from hydra.utils import instantiate
import torch
from torch import nn
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class MLMModule(ptlsMLMPretrainModule):
    def __init__(
        self,
        encoder: DictConfig,
        decoder: Optional[DictConfig] = None,
        **kwargs,
    ):
        self.encoder: SeqEncoderContainer = instantiate(encoder)
        self.decoder: nn.Module = instantiate(decoder) if decoder else nn.Identity()
        super().__init__(self.encoder.trx_encoder, self.encoder.seq_encoder, **kwargs)

    def forward(self, z: PaddedBatch):
        embs = super().forward(z)
        return self.decoder(embs)
