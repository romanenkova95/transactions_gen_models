from typing import Optional
import torch
from torch import nn
from ptls.nn import RnnSeqEncoder


class PretrainedRnnSeqEncoder(RnnSeqEncoder):
    """Pretrained network layer which makes representation for single transactions

    Args:
        path_to_dict (str): Path to state dict of a pretrained RnnSeqEncoder
        **seq_encoder_params: Params for RnnSeqEncoder initialization
    """

    def __init__(self, path_to_dict: Optional[str]=None, freeze: bool=True, **seq_encoder_params):
        super().__init__(**seq_encoder_params)

        if path_to_dict is not None:
            self.load_state_dict(torch.load(path_to_dict))

        self.freeze = freeze
        if freeze:
            # freeze parameters
            for param in self.parameters():
                param.requires_grad = False

    @property
    def output_size(self) -> int:
        """Returns embedding size of a single transaction"""
        return self.embedding_size

    def train(self, mode: bool = True): # override to disable training when frozen
        if self.freeze:
            mode = False

        return super().train(mode)
