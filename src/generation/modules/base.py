from pathlib import Path
from typing import Any, Iterator, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion


import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load import PaddedBatch

from src.generation.decoders.base import AbsDecoder
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class AbsAE(LightningModule):
    """Abstract autoencoder class.

    Attributes:
        lr (float): The learning rate, extracted from the optimizer_config.
        ae_output_size (int): The output size of the decoder.
    """

    def __init__(
        self,
        encoder_config: DictConfig,
        decoder_config: DictConfig,
        mcc_vocab_size: int,
        encoder_weights: Optional[str] = None,
        decoder_weights: Optional[str] = None,
        unfreeze_enc_after: Optional[int] = None,
        unfreeze_dec_after: Optional[int] = None,
    ) -> None:
        """Initialize AbsAE internal state

        Args:
            encoder_config (DictConfig):
                An instantiate-compatible config of the encoder.
            decoder_config (DictConfig):
                An instantiate-compatible config of the decoder.
            mcc_vocab_size (int):
                Total amount of mcc codes (except padding).
            encoder_weights (Optional[str], optional):
                Path to encoder weights. If not specified, no weights are loaded by default.
            decoder_weights (Optional[str], optional):
                Path to decoder weights. If not specified, no weights are loaded by default.
            unfreeze_enc_after (Optional[int], optional):
                Number of epochs to wait before unfreezing encoder weights.
                The module doesn't get frozen by default.
                A negative number would freeze the weights indefinetly.
            unfreeze_dec_after (Optional[int], optional):
                Number of epochs to wait before unfreezing encoder weights.
                The module doesn't get frozen by default.
                A negative number would freeze the weights indefinetly.
        """
        super().__init__()

        self.encoder: SeqEncoderContainer = instantiate(encoder_config)
        self.decoder: AbsDecoder = instantiate(decoder_config)

        self.mcc_vocab_size = mcc_vocab_size
        self.unfreeze_enc_after = unfreeze_enc_after
        self.unfreeze_dec_after = unfreeze_dec_after
        self.ae_output_size = self.decoder.output_size

        if encoder_weights:
            self.encoder.load_state_dict(torch.load(encoder_weights))

        if decoder_weights:
            self.decoder.load_state_dict(torch.load(decoder_weights))

        if unfreeze_enc_after:
            logger.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        if unfreeze_dec_after:
            logger.info("Freezing decoder weights")
            self.decoder.requires_grad_(False)

    def forward(
        self, 
        x: PaddedBatch, 
        return_latent=False
        ) -> Union[Tensor, tuple[Tensor, Union[PaddedBatch, Tensor]]]:
        """Run the forward pass, passing x through encoder and decoder.

        Args:
            x (PaddedBatch): Input, raw transactional data.
            return_latent (bool):  whether to return latent embeddings, decoder output.

        Raises:
            ValueError:
            Unknown format of encoder output. 

        Returns:
            Union[Tensor, tuple[Tensor, Tensor]]: 
                output embeddings, of shape (batch_size, sequence_length, ae_output_size),
                and, possibly, latent embeddings, outputted by the encoder.
                
        Notes:
            The encoder should output one of the following:
                - 3D Tensor of shape (B, L, C), sequence of embeddings.
                - PaddedBatch of shape (B, L, C), sequence of embeddings.
                - 2D Tensor of shape (B, C), a single encoded embedding for each batch element.
        """
        latent_embeddings: Union[PaddedBatch, Tensor] = self.encoder(x)

        # Determine kind of embeddings
        if isinstance(latent_embeddings, PaddedBatch):
            # PaddedBatch embeddings, seq2seq task
            out_embeddings = self.decoder(latent_embeddings.payload)
        elif latent_embeddings.ndim == 3:
            # Embeddings have shape (B, L, E), seq2seq task
            out_embeddings = self.decoder(latent_embeddings)
        elif latent_embeddings.ndim == 2:
            # Embeddings have shape (B, E), nlp generative task; pass sequence length as parameter
            out_embeddings = self.decoder(latent_embeddings, x.seq_feature_shape[1])
        else:
            raise ValueError(f"Unsupported embeddings returned by encoder")
        
        if return_latent:
            return out_embeddings, latent_embeddings
        else:
            return out_embeddings

    def on_train_epoch_start(self) -> None:
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.encoder.requires_grad_(True)

        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")

            self.decoder.requires_grad_(True)
            self.parameters()

        return super().on_train_epoch_start()

    @staticmethod
    def _parse_optimizer_config(
        optimizer_config: DictConfig, params: Iterator[Parameter]
    ):
        optimizer = instantiate(optimizer_config["optimizer"], params=params)
        scheduler = optimizer_config.get("lr_scheduler")
        if scheduler:
            if isinstance(scheduler, dict):
                scheduler = instantiate(scheduler, scheduler={"optimizer": optimizer})
            else:
                scheduler = instantiate(scheduler, optimizer=optimizer)

            return {"lr_scheduler": scheduler, "optimizer": optimizer}
        else:
            return {"optimizer": optimizer}

    # Overriding lr_scheduler_step to fool the exception (which doesn't appear in later versions of pytorch_lightning):
    # pytorch_lightning.utilities.exceptions.MisconfigurationException:
    #   The provided lr scheduler `...` doesn't follow PyTorch's LRScheduler API.
    #   You should override the `LightningModule.lr_scheduler_step` hook with your own logic if you are using a custom LR scheduler.
    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric
    ) -> None:
        return super().lr_scheduler_step(scheduler, optimizer_idx, metric)
