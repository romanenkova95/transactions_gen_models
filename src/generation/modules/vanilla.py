from pathlib import Path
from typing import Literal, Optional, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import LightningModule

import torch
from torch import nn, Tensor

from hydra.utils import instantiate
from torchmetrics.functional import auroc, f1_score, r2_score, average_precision

from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from src.generation.decoders.base import AbsDecoder
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class VanillaAE(LightningModule):
    """A vanilla autoencoder, without masking, just encodes target sequence and then restores it.
    Logs train/val/test losses:
     - a CrossEntropyLoss on mcc codes
     - an MSELoss on amounts
    and train/val/test metrics:
     - a macro-averaged multiclass f1-score on mcc codes
     - a macro-averaged multiclass auroc score on mcc codes
     - an r2-score on amounts

     Attributes:
        out_amount (nn.Linear):
            A linear layer, which restores the transaction amounts.
        out_mcc (nn.Linear):
            A linear layer, which restores the transaction mcc codes.
        amount_loss_weight (float):
            Normalized loss weight for the transaction amount MSE loss.
        mcc_loss_weight (float):
            Normalized loss weight for the transaction mcc code CE loss.
        lr (float):
            The learning rate, extracted from the optimizer_config.
        ae_output_size (int):
            The output size of the decoder.

    Notes:
        amount_loss_weight, mcc_loss_weight are normalized so that amount_loss_weight + mcc_loss_weight = 1.
        This is done to remove one hyperparameter. Loss gradient size can be managed separately through lr.

    """

    def __init__(
        self,
        loss_weights: dict[Literal["amount", "mcc"], float],
        encoder: DictConfig,
        decoder: DictConfig,
        mcc_head: DictConfig,
        amount_head: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
        encoder_weights: Optional[str] = None,
        decoder_weights: Optional[str] = None,
        unfreeze_enc_after: Optional[int] = None,
        unfreeze_dec_after: Optional[int] = None,
    ) -> None:
        """Initialize VanillaAE internal state.

        Args:
            loss_weights (dict):
                A dictionary with keys "amount" and "mcc", mapping them to the corresponding loss weights
            encoder (SeqEncoderContainer):
                SeqEncoderContainer to be used as an encoder.
            decoder (AbsDecoder):
                AbsDecoder, to be used as the decoder.
            mcc_head (DictConfig):
                DictConfig for mcc head, instantiated with in_channels keyword argument.
            amount_head (DictConfig):
                Partial dictconfig for amount head, instantiated with in_channels keyword argument.
            optimizer (DictConfig):
                Optimizer dictconfig, instantiated with params kwarg.
            scheduler (Optional[DictConfig]):
                Optionally, an lr scheduler dictconfig, instantiated with optimizer kwarg
            scheduler_config (Optional[dict]):
                An lr_scheduler config for specifying scheduler-specific params, such as which metric to monitor
                See LightningModule.configure_optimizers docstring for more details.
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
        self.save_hyperparameters()

        self.encoder: SeqEncoderContainer = instantiate(encoder)
        self.decoder: AbsDecoder = instantiate(decoder)

        self.unfreeze_enc_after = unfreeze_enc_after
        self.unfreeze_dec_after = unfreeze_dec_after
        self.ae_output_size = self.decoder.output_size

        if encoder_weights:
            self.encoder.load_state_dict(torch.load(Path(encoder_weights)))

        if decoder_weights:
            self.decoder.load_state_dict(torch.load(Path(decoder_weights)))

        if unfreeze_enc_after:
            logger.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        if unfreeze_dec_after:
            logger.info("Freezing decoder weights")
            self.decoder.requires_grad_(False)

        self.amount_head = instantiate(amount_head, in_channels=self.ae_output_size)

        self.mcc_head = instantiate(mcc_head, in_channels=self.ae_output_size)

        self.optimizer_dictconfig = optimizer
        self.scheduler_dictconfig = scheduler
        self.scheduler_config = scheduler_config or {}

        self.amount_loss_weight = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weight = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

    def on_train_epoch_start(self) -> None:
        """Overrided method to unfreeze encoder/decoder weights"""
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.encoder.requires_grad_(True)

        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")

            self.decoder.requires_grad_(True)
            self.parameters()

        return super().on_train_epoch_start()

    def forward(
        self,
        batch: PaddedBatch,
    ) -> tuple[Tensor, Tensor, Union[PaddedBatch, Tensor]]:
        """Run the forward pass of the VanillaAE module.
        Pass the batch through the autoencoder, and afterwards pass it through mcc_head & amount_head.
        to get the respective targets.

        Args:
            batch (PaddedBatch): Input batch of raw transactional data.
            return_latent (bool): whether to return latent embeddings (the ones after encoder)

        Returns:
            tuple[Tensor, Tensor]:
                tuple of tensors:
                    - Predicted mcc logits, shape (B, L, mcc_vocab_size + 1)
                    - predicted amounts, shape (B, L)
                    - Latent embeddings

        Notes:
            The padding elements, determined by the padding mask of the input PaddedBatch,
            are zeroed out to prevent gradient flow.

        """
        latent_embeddings: Union[PaddedBatch, Tensor] = self.encoder(batch)

        if self.encoder.is_reduce_sequence:
            # Encoder returned batch of single-vector embeddings (one per input sequence),
            # need to pass shape for decoder to construct output sequence
            seqs_after_lstm = self.decoder(
                latent_embeddings, batch.seq_feature_shape[1]
            )
        else:
            # Encoder returned PaddedBatch of embeddings
            seqs_after_lstm = self.decoder(latent_embeddings.payload) # type: ignore

        mcc_pred: Tensor = self.mcc_head(seqs_after_lstm)
        amount_pred: Tensor = self.amount_head(seqs_after_lstm).squeeze(dim=-1)

        # zero-out padding to disable grad flow
        pad_mask = batch.seq_len_mask.bool().reshape(*(amount_pred.shape))
        mcc_pred[~pad_mask] = 0
        amount_pred[~pad_mask] = 0

        return mcc_pred, amount_pred, latent_embeddings

    def _calculate_metrics(
        self,
        mcc_preds: Tensor,
        amt_value: Tensor,
        mcc_target: Tensor,
        amt_target: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the metrics

        Args:
            mcc_preds (Tensor): predicted mcc logits, (B, L, mcc_vocab_size)
            amt_value (Tensor): predicted amounts, (B, L)
            mcc_target (Tensor): target mccs, (B, L)
            amt_target (Tensor): target amounts, (B, L)
            mask (Tensor): mask of non-padding elements

        Returns:
            dict[str, Tensor]: Dictionary of metrics, with keys mcc_auroc, mcc_f1, amt_r2
        """
        with torch.no_grad():
            mcc_target = mcc_target[mask]
            mcc_preds = mcc_preds[mask].reshape((*mcc_target.shape, -1))

            labels = mcc_target.unique()
            num_classes = len(labels)
            mcc_target = torch.argwhere(mcc_target[:, None] == labels[None, :])[:, 1]
            mcc_preds = mcc_preds[:, labels]

            return {
                "mcc_auroc": auroc(
                    mcc_preds,
                    mcc_target,
                    average="macro",
                    num_classes=num_classes,
                    task="multiclass",
                ).cpu(),  # type: ignore
                "mcc_prauc": average_precision(
                    mcc_preds,
                    mcc_target,
                    average="macro",
                    num_classes=num_classes,
                    task="multiclass",
                ),  # type: ignore
                "mcc_f1": f1_score(
                    mcc_preds,
                    mcc_target,
                    average="macro",
                    num_classes=num_classes,
                    task="multiclass",
                ).cpu(),
                "amt_r2": r2_score(
                    amt_value[mask],
                    amt_target[mask],
                ).cpu(),
            }

    def _calculate_losses(
        self,
        mcc_pred: Tensor,
        amount_pred: Tensor,
        mcc_target: Tensor,
        amount_target: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the losses, weigh them with respective weights

        Args:
            mcc_pred (Tensor): Predicted mcc logits, (B, L, mcc_vocab_size).
            amount_pred (Tensor): Predicted amounts, (B, L).
            mcc_target (Tensor): target mcc codes.
            amount_target (Tensor): target amounts.

        Returns:
            Dictionary of losses, with keys loss, loss_mcc, loss_amt.
        """
        mcc_loss = self.mcc_criterion(mcc_pred.transpose(2, 1), mcc_target)
        amount_loss = self.amount_criterion(amount_pred, amount_target)

        total_loss = (
            self.mcc_loss_weight * mcc_loss + self.amount_loss_weight * amount_loss
        )

        return {"loss": total_loss, "loss_mcc": mcc_loss, "loss_amt": amount_loss}

    def _step(
        self,
        stage: str,
        batch: PaddedBatch,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Generalized function to do a train/val/test step.

        Args:
            stage (str): train, val, or test, depending on the stage.
            batch (PaddedBatch): Input.
            batch_idx (int): ignored

        Returns:
            STEP_OUTPUT:
                if stage == "train", returns total loss.
                else returns a dictionary of metrics.
        """
        mcc_pred, amount_pred, _ = self(batch)  # (B * S, L, MCC_N), (B * S, L)
        mcc_target = batch.payload["mcc_code"]
        amount_target = torch.log(batch.payload["amount"] + 1)  # Logarithmize targets

        loss_dict = self._calculate_losses(
            mcc_pred, amount_pred, mcc_target, amount_target
        )

        metric_dict = self._calculate_metrics(
            mcc_pred, amount_pred, mcc_target, amount_target, batch.seq_len_mask.bool()
        )

        self.log_dict(
            {f"{stage}_{k}": v for k, v in loss_dict.items()},
            on_step=(stage == "train"),
            on_epoch=(stage != "train"),
            batch_size=batch.seq_feature_shape[0],
        )

        self.log_dict(
            {f"{stage}_{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.seq_feature_shape[0],
        )

        if stage == "train":
            return loss_dict["loss"]
        else:
            return metric_dict

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("test", *args, **kwargs)

    def predict_step(
        self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run the predict step: forward pass for the input batch, and trim padding in output.

        Args:
            batch (PaddedBatch): input padded batch
            batch_idx (int): ignored
            dataloader_idx (int, optional): ignored

        Returns:
            tuple[list[Tensor], list[Tensor]]:
                - list of predicted mcc logits, (B, L_i, mcc_vocab_size)
                - list of predicted amounts, (B, L_i)
                Note that L_i (i=0...B-1) is different for each element of the batch,
                for this reason we return a list and not a tensor.
        """
        mcc_pred: Tensor  # (B, L, MCC_NUM)
        amount_pred: Tensor  # (B, L)
        mcc_pred, amount_pred = self(batch)
        lens_mask = batch.seq_len_mask.bool()
        lens = batch.seq_lens

        mcc_pred_trim = mcc_pred[lens_mask]
        amount_pred_trim = amount_pred[lens_mask]

        return mcc_pred_trim.split(lens), torch.exp(amount_pred_trim.split(lens)) # type: ignore

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_dictconfig, params=self.parameters())

        if self.scheduler_dictconfig:
            scheduler = instantiate(self.scheduler_dictconfig, optimizer=optimizer)
            scheduler_config = {"scheduler": scheduler, **self.scheduler_config}

            return [optimizer], [scheduler_config]

        return optimizer

    # Overriding lr_scheduler_step to fool the exception (which doesn't appear in later versions of pytorch_lightning):
    # pytorch_lightning.utilities.exceptions.MisconfigurationException:
    #   The provided lr scheduler `...` doesn't follow PyTorch's LRScheduler API.
    #   You should override the `LightningModule.lr_scheduler_step` hook with your own logic if you are using a custom LR scheduler.
    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric
    ) -> None:
        return super().lr_scheduler_step(scheduler, optimizer_idx, metric)
