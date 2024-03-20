"""The file with the main logic for the AR model."""

from typing import Literal, Optional, Union

import torch
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from src.utils.logging_utils import get_logger

from .vanilla import VanillaAE

logger = get_logger(name=__name__)


class ARModule(VanillaAE):
    """A module for AR training, just encodes the sequence and predicts its shifted version.

    Logs train/val/test losses:
     - a CrossEntropyLoss on mcc codes
     - an MSELoss on amounts
    and train/val/test metrics:
     - a macro-averaged multiclass f1-score on mcc codes
     - a macro-averaged multiclass auroc score on mcc codes
     - an r2-score on amounts.

    Attributes
    ----------
        amount_loss_weight (float):
            Normalized loss weight for the transaction amount MSE loss.
        mcc_loss_weight (float):
            Normalized loss weight for the transaction mcc code CE loss.
        lr (float):
            The learning rate, extracted from the optimizer_config.

    Notes
    -----
        amount_loss_weight, mcc_loss_weight are normalized so that amount_loss_weight + mcc_loss_weight = 1.
        This is done to remove one hyperparameter. Loss gradient size can be managed separately through lr.

    """

    def __init__(
        self,
        loss_weights: dict[Literal["amount", "mcc"], float],
        encoder: DictConfig,
        optimizer: DictConfig,
        num_types: Optional[int],
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
        encoder_weights: Optional[str] = None,
        freeze_enc: Optional[bool] = False,
    ) -> None:
        """Initialize GPTModule internal state.

        Args:
        ----
            loss_weights (dict):
                A dictionary with keys "amount" and "mcc", mapping them to the corresponding loss weights
            encoder (SeqEncoderContainer):
                SeqEncoderContainer to be used as an encoder.
            num_types (int):
                Amount of mcc types; clips all input to this value.
            optimizer (DictConfig):
                Optimizer dictconfig, instantiated with params kwarg.
            scheduler (Optional[DictConfig]):
                Optionally, an lr scheduler dictconfig, instantiated with optimizer kwarg
            scheduler_config (Optional[dict]):
                An lr_scheduler config for specifying scheduler-specific params, such as which metric to monitor
                See LightningModule.configure_optimizers docstring for more details.
            encoder_weights (Optional[str], optional):
                Path to encoder weights. If not specified, no weights are loaded by default.
            freeze_enc (Optional[int], optional):
                Whether to freeze the encoder module.
        """
        super().__init__(
            loss_weights=loss_weights,
            encoder=encoder,
            optimizer=optimizer,
            num_types=num_types,  # type: ignore
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            encoder_weights=encoder_weights,
            freeze_enc=freeze_enc,
        )

        self.encoder.is_reduce_sequence = False  # AR is trained in seq2seq regime

    def forward(self, x):
        """Encode the data in x."""
        return self.encoder(x)

    def shared_step(
        self,
        stage: Literal["train", "val", "test"],
        batch: PaddedBatch,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Generalized function to do a train/val/test step.

        Args:
        ----
            stage (str): train, val, or test, depending on the stage.
            batch (PaddedBatch): Input.
            batch_idx (int): ignored.
            *args: ignored.
            **kwargs: ignored.

        Returns:
        -------
            STEP_OUTPUT:
                if stage == "train", returns total loss.
                else returns a dictionary of metrics.
        """
        batch.payload["mcc_code"] = torch.clip(
            batch.payload["mcc_code"], 0, self.num_types - 1
        )

        embeddings = self(batch).payload

        mcc_pred = self.mcc_head(embeddings)[:, :-1, :]
        amount_pred = self.amount_head(embeddings)[:, :-1].squeeze(-1)

        mcc_target = batch.payload["mcc_code"][:, 1:]
        amount_target = batch.payload["amount"][:, 1:]
        amount_target = (
            amount_target.abs().log1p() * amount_target.sign()
        )  # Logarithmize targets

        nonpad_mask = batch.seq_len_mask[:, 1:].bool()

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
                metric,  # type: ignore
                on_step=False,
                on_epoch=True,
                batch_size=batch.seq_feature_shape[0],
            )

        return loss_dict["loss"]

    def predict_step(
        self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Union[tuple[list[Tensor], list[Tensor]], tuple[Tensor, Tensor]]:
        """Run self on input batch."""
        return self(batch)
