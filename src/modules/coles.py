"""CoLES model"""
from omegaconf import DictConfig
from hydra.utils import instantiate

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.frames.coles import CoLESModule


class CustomCoLES(CoLESModule):
    """
    Custom coles module inhereted from ptls coles module.
    """

    def __init__(
        self,
        optimizer_partial: DictConfig,
        lr_scheduler_partial: DictConfig,
        encoder: DictConfig,
    ) -> None:
        """Overrided initialize method, which is suitable for our tasks

        Args:
            optimizer_partial (DictConfig with Callable):
                Partial initialized torch optimizer (with parameters)
            lr_scheduler_partial (DictConfig with Callable):
                Partial initialized torch lr scheduler (with parameters)
            encoder (DictConfig with SeqEncoderContainer): Ptls sequence encoder
                (including sequence encoder and single transaction encoder)
        """
        self.save_hyperparameters()
        enc: SeqEncoderContainer = instantiate(encoder)
        super().__init__(
            seq_encoder=enc,
            optimizer_partial=instantiate(optimizer_partial, _partial_=True),
            lr_scheduler_partial=instantiate(lr_scheduler_partial, _partial_=True),
        )

        self.encoder = enc
