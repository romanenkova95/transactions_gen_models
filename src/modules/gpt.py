from pathlib import Path
from typing import Any, Literal, Optional, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn, Tensor

from hydra.utils import instantiate

from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss

from src.utils.logging_utils import get_logger

from .vanilla import VanillaAE


logger = get_logger(name=__name__)


class GPTModule(VanillaAE):
    """A module for GPT-like training, just encodes the sequence and predicts its shifted version.
    Logs train/val/test losses:
     - a CrossEntropyLoss on mcc codes
     - an MSELoss on amounts
    and train/val/test metrics:
     - a macro-averaged multiclass f1-score on mcc codes
     - a macro-averaged multiclass auroc score on mcc codes
     - an r2-score on amounts

     Attributes:
        amount_loss_weight (float):
            Normalized loss weight for the transaction amount MSE loss.
        mcc_loss_weight (float):
            Normalized loss weight for the transaction mcc code CE loss.
        lr (float):
            The learning rate, extracted from the optimizer_config.

    Notes:
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
            num_types=num_types,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            encoder_weights=encoder_weights,
            freeze_enc=freeze_enc,
        )

    def forward(self, x):
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
            stage (str): train, val, or test, depending on the stage.
            batch (PaddedBatch): Input.
            batch_idx (int): ignored

        Returns:
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
        return self(batch)


class GPTContrastiveModule(VanillaAE):
    """A module for GPT-like contrastive training, just encodes the sequence and predicts the embeddings of its shifted version.
    Logs train/val/test losses:
     - QuerySoftmaxLoss

     Attributes:
        lr (float):
            The learning rate, extracted from the optimizer_config.

    Notes:
        Loss gradient size can be managed separately through lr.

    """

    def __init__(
        self,
        encoder: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
        encoder_weights: Optional[str] = None,
        freeze_enc: Optional[bool] = False,
        neg_count: int = 5,
        temperature: float = 20.0,
    ) -> None:
        """Initialize GPTContrastiveModule internal state.

        Args:
            encoder (DictConfig):
                SeqEncoderContainer dictconfig to be used as an encoder.
            optimizer (DictConfig):
                Optimizer dictconfig, instantiated with params kwarg.
            scheduler (Optional[DictConfig]):
                Optionally, an lr scheduler dictconfig, instantiated with optimizer kwarg
            scheduler_config (Optional[dict]):
                An lr_scheduler config for specifying scheduler-specific params, such as which metric to monitor
                See LightningModule.configure_optimizers docstring for more details.
            neg_count (int):
                negative count for `QuerySoftmaxLoss`
            temperature (float):
                temperature parameter of `QuerySoftmaxLoss`
        """
        super(VanillaAE, self).__init__()

        self.save_hyperparameters()

        self.encoder: SeqEncoderContainer = instantiate(encoder)
        self.head = nn.Linear(
            self.encoder.embedding_size, self.encoder.trx_encoder.output_size
        )

        self.optimizer_dictconfig = optimizer
        self.scheduler_dictconfig = scheduler
        self.scheduler_config = scheduler_config or {}

        if encoder_weights:
            self.encoder.load_state_dict(torch.load(Path(encoder_weights)))

        self.freeze_enc = freeze_enc
        if freeze_enc:
            logger.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        self.loss_fn = QuerySoftmaxLoss(temperature, reduce=True)

    def forward(self, x):
        return self.encoder(x)

    def get_neg_ix(self, mask):
        """Sample from predicts, where `mask == True`, without self element.
        sample from predicted tokens from batch
        """
        mask_num = mask.int().sum()
        mn = 1 - torch.eye(mask_num, device=mask.device)
        neg_ix = torch.multinomial(mn, self.hparams.neg_count)

        b_ix = (
            torch.arange(mask.size(0), device=mask.device)
            .view(-1, 1)
            .expand_as(mask)[mask][neg_ix]
        )
        t_ix = (
            torch.arange(mask.size(1), device=mask.device)
            .view(1, -1)
            .expand_as(mask)[mask][neg_ix]
        )
        return b_ix, t_ix

    def shared_step(
        self,
        stage,
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
            loss: computed value of the loss function
        """
        mask = batch.seq_len_mask.bool()[:, 1:]
        x_trx = self.encoder.trx_encoder(batch).payload[:, 1:]

        embeddings = self(batch).payload[:, :-1]
        out = self.head(embeddings)

        target = x_trx[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H

        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H

        loss = self.loss_fn(target, predict, negative)

        self.log_dict(
            {f"{stage}_loss": loss.item()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.seq_feature_shape[0],
        )

        return loss

    def predict_step(
        self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Union[tuple[list[Tensor], list[Tensor]], tuple[Tensor, Tensor]]:
        return self(batch)
