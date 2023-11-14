from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import LightningModule
from sklearn import multiclass

import torch
from torch import nn, Tensor

from hydra.utils import instantiate
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Metric,
    MetricCollection,
    MultitaskWrapper,
    R2Score,
)
from torchmetrics.functional import auroc, f1_score, r2_score, average_precision

from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from src.nn.decoders.base import AbsDecoder
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
        mcc_head: DictConfig,
        amount_head: DictConfig,
        optimizer: DictConfig,
        decoder: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
        encoder_weights: Optional[str] = None,
        freeze_enc: Optional[bool] = False,
        reconstruction_len: Optional[int] = None,
    ) -> None:
        """Initialize VanillaAE internal state.

        Args:
            loss_weights (dict):
                A dictionary with keys "amount" and "mcc", mapping them to the corresponding loss weights
            encoder (SeqEncoderContainer):
                SeqEncoderContainer to be used as an encoder.
            mcc_head (DictConfig):
                DictConfig for mcc head, instantiated with in_channels keyword argument.
            amount_head (DictConfig):
                Partial dictconfig for amount head, instantiated with in_channels keyword argument.
            optimizer (DictConfig):
                Optimizer dictconfig, instantiated with params kwarg.
            decoder (AbsDecoder):
                AbsDecoder, to be used as the decoder.
            scheduler (Optional[DictConfig]):
                Optionally, an lr scheduler dictconfig, instantiated with optimizer kwarg
            scheduler_config (Optional[dict]):
                An lr_scheduler config for specifying scheduler-specific params, such as which metric to monitor
                See LightningModule.configure_optimizers docstring for more details.
            encoder_weights (Optional[str], optional):
                Path to encoder weights. If not specified, no weights are loaded by default.
            freeze_enc (Optional[int], optional):
                Whether to freeze the encoder module.
            reconstruction_len (Optional[int]):
                length of reconstructed batch in predict_step, optional.
                If None, determine length from batch.seq_lens.
                If int, reconstruct that many tokens.
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder: SeqEncoderContainer = instantiate(encoder)
        self.decoder: AbsDecoder = (
            instantiate(decoder) if decoder else AbsDecoder(self.encoder.embedding_size)
        )

        self.ae_output_size = self.decoder.output_size

        if encoder_weights:
            self.encoder.load_state_dict(torch.load(Path(encoder_weights)))

        self.freeze_enc = freeze_enc
        if freeze_enc:
            logger.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        self.amount_head = instantiate(amount_head, in_channels=self.ae_output_size)
        self.mcc_head = instantiate(mcc_head, in_channels=self.ae_output_size)

        self.optimizer_dictconfig = optimizer
        self.scheduler_dictconfig = scheduler
        self.scheduler_config = scheduler_config or {}

        self.amount_loss_weight = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weight = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

        self.reconstruction_len = reconstruction_len

        multiclass_args: dict[str, Any] = dict(
            task="multiclass",
            num_classes=self.mcc_head[-2].out_features,
            ignore_index=0,
        )

        MetricsType = dict[Literal["mcc", "amount"], MetricCollection]
        def make_metrics(stage: str) -> MetricsType:
            return nn.ModuleDict({
                "mcc": MetricCollection(
                    AUROC(**multiclass_args, average="weighted"),
                    F1Score(**multiclass_args, average="micro"),
                    AveragePrecision(**multiclass_args, average="weighted"),
                    Accuracy(**multiclass_args, average="micro"),
                    prefix=stage
                ),
                "amount": MetricCollection(R2Score(), prefix=stage),
            }) # type: ignore

        self.train_metrics: MetricsType = make_metrics("train")
        self.val_metrics: MetricsType = make_metrics("val")
        self.test_metrics: MetricsType = make_metrics("test")

    @property
    def metric_name(self):
        return "val_loss"
    
    def train(self, mode: bool = True): # eval encoder if needed
        super().train(mode)
        if self.freeze_enc:
            self.encoder.eval()
            
        return self

    def forward(
        self, batch: PaddedBatch, L: Optional[int] = None
    ) -> tuple[Tensor, Tensor, Union[PaddedBatch, Tensor], Tensor]:
        """Run the forward pass of the VanillaAE module.
        Pass the batch through the autoencoder, and afterwards pass it through mcc_head & amount_head.
        to get the respective targets.

        Args:
            batch (PaddedBatch): Input batch of raw transactional data.
            L (int): Optionally, specify length of decoded sequence

        Returns:
            tuple[Tensor, Tensor]:
                tuple of tensors:
                    - Predicted mcc logits, shape (B, L, mcc_vocab_size + 1)
                    - predicted amounts, shape (B, L)
                    - Latent embeddings
                    - Pad mask

        Notes:
            The padding elements, determined by the padding mask of the input PaddedBatch,
            are zeroed out to prevent gradient flow.

        """
        latent_embeddings: Union[PaddedBatch, Tensor] = self.encoder(batch)

        L = L or batch.seq_feature_shape[1]
        if self.encoder.is_reduce_sequence:
            # Encoder returned batch of single-vector embeddings (one per input sequence),
            # need to pass shape for decoder to construct output sequence
            seqs_after_lstm = self.decoder(latent_embeddings, L)
        else:
            # Encoder returned PaddedBatch of embeddings
            seqs_after_lstm = self.decoder(latent_embeddings.payload)  # type: ignore

        mcc_pred: Tensor = self.mcc_head(seqs_after_lstm)
        amount_pred: Tensor = self.amount_head(seqs_after_lstm).squeeze(dim=-1)

        # mask to calculate losses & metrics on
        nonpad_mask = batch.seq_len_mask.bool()
        return mcc_pred, amount_pred, latent_embeddings, nonpad_mask

    def _calculate_losses(
        self,
        mcc_pred: Tensor,
        amount_pred: Tensor,
        mcc_target: Tensor,
        amount_target: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the losses, weigh them with respective weights

        Args:
            mcc_pred (Tensor): Predicted mcc logits, (B, L, mcc_vocab_size).
            amount_pred (Tensor): Predicted amounts, (B, L).
            mcc_target (Tensor): target mcc codes.
            amount_target (Tensor): target amounts.
            mask (Tensor): mask of non-padding elements

        Returns:
            Dictionary of losses, with keys loss, loss_mcc, loss_amt.
        """
        mcc_loss = self.mcc_criterion(mcc_pred[mask], mcc_target[mask])
        amount_loss = self.amount_criterion(amount_pred[mask], amount_target[mask])

        total_loss = (
            self.mcc_loss_weight * mcc_loss + self.amount_loss_weight * amount_loss
        )

        return {"loss": total_loss, "loss_mcc": mcc_loss, "loss_amt": amount_loss}

    def shared_step(
        self,
        stage: Literal["train", "val", "test"],
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
        mcc_pred, amount_pred, _, nonpad_mask = self(
            batch
        )  # (B * S, L, MCC_N), (B * S, L)
        mcc_target = batch.payload["mcc_code"]
        amount_target = torch.log(batch.payload["amount"] + 1)  # Logarithmize targets

        loss_dict = self._calculate_losses(
            mcc_pred, amount_pred, mcc_target, amount_target, nonpad_mask
        )

        metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }[stage]

        metrics["mcc"].update(mcc_pred[nonpad_mask], mcc_target[nonpad_mask])
        metrics["amount"].update(amount_pred[nonpad_mask], amount_target[nonpad_mask])

        self.log_dict(
            {f"{stage}_{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.seq_feature_shape[0],
        )
        
        for metric in metrics.values():
            self.log_dict(
                metric, # type: ignore
                on_step=False,
                on_epoch=True,
                batch_size=batch.seq_feature_shape[0],
            )

        return loss_dict["loss"]

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.shared_step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self.shared_step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self.shared_step("test", *args, **kwargs)

    def predict_step(
        self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Union[tuple[list[Tensor], list[Tensor]], tuple[Tensor, Tensor]]:
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
        mcc_pred, amount_pred, _, _ = self(batch, self.reconstruction_len)
        if self.reconstruction_len:
            return mcc_pred, torch.exp(amount_pred)
        else:
            lens_mask = batch.seq_len_mask.bool()
            lens = batch.seq_lens.tolist()

            mcc_pred_trim = mcc_pred[lens_mask]
            amount_pred_trim = amount_pred[lens_mask]

            return mcc_pred_trim.split(lens), torch.exp(amount_pred_trim).split(lens)

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
