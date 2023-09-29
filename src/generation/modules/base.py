from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion


import torch
from torch import Tensor

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
        optimizer_config: DictConfig,
        encoder_weights: Optional[str] = "",
        decoder_weights: Optional[str] = "",
        unfreeze_enc_after: Optional[int] = 0,
        unfreeze_dec_after: Optional[int] = 0,
    ) -> None:
        """Initialize AbsAE internal state

        Args:
            encoder_config (DictConfig): 
                An instantiate-compatible config of the encoder.
            decoder_config (DictConfig): 
                An instantiate-compatible config of the decoder.
            mcc_vocab_size (int): 
                Total amount of mcc codes (except padding).
            optimizer_config (DictConfig): 
                A dict config with an optimizer key and optionally an lr_scheduler key.
                Both optimizer & scheduler are partially instantiated, and then initialized with
                model parameters & optimizer respectfully. 
                lr_scheduler may be either a torch lr_scheduler instance, 
                or a dict with lr_scheduler config (see configure_optimizers docs).
            encoder_weights (Optional[str], optional): 
                Path to encoder weights. Defaults to "", in which case no weights are loaded.
            decoder_weights (Optional[str], optional): 
                Path to decoder weights. Defaults to "", in which case no weights are loaded.
            unfreeze_enc_after (Optional[int], optional): 
                Number of epochs to wait before unfreezing encoder weights. 
                Defaults to 0, in which case the module isn't frozen.
                A negative number would freeze the weights for the whole training duration.
            unfreeze_dec_after (Optional[int], optional): 
                Number of epochs to wait before unfreezing encoder weights. 
                Defaults to 0, in which case the module isn't frozen.
                A negative number would freeze the weights for the whole training duration.
        """
        super().__init__()

        self.encoder: SeqEncoderContainer = instantiate(encoder_config)
        self.decoder: AbsDecoder = instantiate(decoder_config)

        self.mcc_vocab_size = mcc_vocab_size
        self.unfreeze_enc_after = unfreeze_enc_after
        self.unfreeze_dec_after = unfreeze_dec_after
        self.ae_output_size = self.decoder.output_size
        self.optimizer_config = optimizer_config
        self.lr = optimizer_config["optimizer"]["lr"]

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

    def forward(self, x: PaddedBatch) -> Any:
        """Run the forward pass, passing x through encoder and decoder.

        Args:
            x (PaddedBatch): Input, presumably raw transactional data.

        Raises:
            ValueError: 
            Unknown format of encoder output. The encoder should output one of the following:
                - 3D Tensor of shape (B, L, C), sequence of embeddings.
                - PaddedBatch of shape (B, L, C), sequence of embeddings.
                - 2D Tensor of shape (B, C), a single encoded embedding for each batch element.

        Returns:
            torch.Tensor: output embeddings, of shape (batch_size, sequence_length, ae_output_size)
        """
        embeddings: Union[PaddedBatch, Tensor] = self.encoder(x)

        # Determine kind of embeddings
        if isinstance(embeddings, PaddedBatch):
            # PaddedBatch embeddings, seq2seq task
            return self.decoder(embeddings.payload)
        elif embeddings.ndim == 3:
            # Embeddings have shape (B, L, E), seq2seq task
            return self.decoder(embeddings)
        elif embeddings.ndim == 2:
            # Embeddings have shape (B, E), nlp generative task; pass sequence length as parameter
            return self.decoder(embeddings, x.seq_feature_shape[1])
        else:
            raise ValueError(f"Unsupported embeddings returned by encoder")

    def on_train_epoch_start(self) -> None:
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.encoder.requires_grad_(True)

        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")
            self.decoder.requires_grad_(True)

        return super().on_train_epoch_start()

    def configure_optimizers(self):
        cnf: Dict = instantiate(self.optimizer_config, _convert_="all")
        cnf["optimizer"] = cnf["optimizer"](
            lr=self.lr, params=self.parameters()
        )

        scheduler = cnf.get("lr_scheduler")
        if scheduler:
            if isinstance(scheduler, dict):
                scheduler["scheduler"] = scheduler["scheduler"](optimizer=cnf["optimizer"])
            else:
                cnf["lr_scheduler"] = scheduler(optimizer=cnf["optimizer"])

        return cnf
    
    # Overriding lr_scheduler_step to fool the exception (which doesn't appear in later versions of pytorch_lightning):
    # pytorch_lightning.utilities.exceptions.MisconfigurationException: 
    #   The provided lr scheduler `...` doesn't follow PyTorch's LRScheduler API. 
    #   You should override the `LightningModule.lr_scheduler_step` hook with your own logic if you are using a custom LR scheduler.
    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric) -> None:
        return super().lr_scheduler_step(scheduler, optimizer_idx, metric)
