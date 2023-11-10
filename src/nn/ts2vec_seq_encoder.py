import numpy as np

import torch
import torch.nn as nn

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch

from .ts2vec_components import DilatedConvEncoder

class ConvEncoder(AbsSeqEncoder):
    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 num_layers=10,
                 dropout=0,
                 is_reduce_sequence=False,  
                 reducer='maxpool'
                 ):
        
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size

        self.feature_extractor = DilatedConvEncoder(
            input_size,
            [input_size] * num_layers + [hidden_size],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(dropout)

        self.reducer = reducer

    def forward(self, x: PaddedBatch):  # x: B x T x input_dims        
        # conv encoder 
        input_ = x.payload.transpose(1, 2)  # B x Ch x T
            
        out = self.repr_dropout(self.feature_extractor(input_))  # B x Co x T
        out = out.transpose(1, 2)  # B x T x Co
        
        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            out = out.payload.max(dim=1).values

        return out


class ConvSeqEncoder(SeqEncoderContainer):
    def __init__(self,
                trx_encoder=None,
                input_size=None,
                is_reduce_sequence=False,
                **seq_encoder_params,
                ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=ConvEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )
