"""CoLES model"""
from typing import Callable

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.frames.coles import CoLESModule


class CustomCoLES(CoLESModule):
    """
    Custom coles module inhereted from ptls coles module.
    """

    def __init__(
        self,
        optimizer_partial: Callable,
        lr_scheduler_partial: Callable,
        sequence_encoder: SeqEncoderContainer
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
            lr_scheduler_partial=lr_scheduler_partial
        )
        self.sequence_encoder_model = sequence_encoder

    def get_seq_encoder_weights(self) -> dict:
        """Get weights of the sequnce encoder in torch format

        Returns:
            dict: Encoder weights
        """
        return self.sequence_encoder_model.state_dict()
