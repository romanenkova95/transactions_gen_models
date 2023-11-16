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
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss

from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from src.nn.decoders.base import AbsDecoder
from src.utils.logging_utils import get_logger


logger = get_logger(name=__name__)


class GPTModule(LightningModule):
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
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
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
            reconstruction_len (Optional[int]):
                length of reconstructed batch in predict_step, optional.
                If None, determine length from batch.seq_lens.
                If int, reconstruct that many tokens.
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder: SeqEncoderContainer = instantiate(encoder)

        self.amount_head = instantiate(amount_head, in_channels=self.encoder.embedding_size)

        self.mcc_head = instantiate(mcc_head, in_channels=self.encoder.embedding_size)

        self.optimizer_dictconfig = optimizer
        self.scheduler_dictconfig = scheduler
        self.scheduler_config = scheduler_config or {}

        self.amount_loss_weight = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weight = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

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

    def forward(self, x):
        return self.encoder(x)

    @property
    def metric_name(self):
        return "val_loss"

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

        embeddings = self(batch).payload

        mcc_pred = self.mcc_head(embeddings)[:, :-1, :]
        amount_pred = self.amount_head(embeddings)[:, :-1].squeeze(-1)
        
        mcc_target = batch.payload["mcc_code"][:, 1:]
        amount_target = torch.log(batch.payload["amount"][:, 1:] + 1)  # Logarithmize targets

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
    

class GPTContrastiveModule(LightningModule):
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
        encoder: DictConfig,
        optimizer: DictConfig, 
        scheduler: Optional[DictConfig] = None,
        scheduler_config: Optional[dict] = None,
        neg_count: int = 5,
        temperature: float = 20,
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
            reconstruction_len (Optional[int]):
                length of reconstructed batch in predict_step, optional.
                If None, determine length from batch.seq_lens.
                If int, reconstruct that many tokens.
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder: SeqEncoderContainer = instantiate(encoder)
        self.head = instantiate({"_target_": "torchvision.ops.MLP"}, in_channels=self.encoder.embedding_size, hidden_channels=[self.encoder.trx_encoder.output_size])

        self.optimizer_dictconfig = optimizer
        self.scheduler_dictconfig = scheduler
        self.scheduler_config = scheduler_config or {}

        self.loss_fn = QuerySoftmaxLoss(temperature, reduce=True)

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, self.encoder.trx_encoder.output_size), requires_grad=True)

    def forward(self, x):
        return self.encoder(x)

    def mask_x(self, x, attention_mask, mask):
        shuffled_tokens = x[attention_mask.bool()]
        B, T, H = x.size()
        ix = torch.multinomial(torch.ones(shuffled_tokens.size(0)), B * T, replacement=True)
        shuffled_tokens = shuffled_tokens[ix].view(B, T, H)

        rand = torch.rand(B, T, device=x.device).unsqueeze(2).expand(B, T, H)
        replace_to = torch.where(
            rand < 0.8,
            self.token_mask.expand_as(x),  # [MASK] token 80%
            shuffled_tokens                # replaced roken                 
        )
        return torch.where(mask.bool().unsqueeze(2).expand_as(x), replace_to, x)

    def get_mask(self, attention_mask):
        last_ind = attention_mask.sum(dim=1) - 1 
        mask = torch.zeros_like(attention_mask)
        mask[:, last_ind] = 1
        return mask.bool()

    def get_neg_ix(self, mask):
        """Sample from predicts, where `mask == True`, without self element.
        sample from predicted tokens from batch
        """
        mask_num = mask.int().sum()
        mn = 1 - torch.eye(mask_num, device=mask.device)
        neg_ix = torch.multinomial(mn, self.hparams.neg_count)

        b_ix = torch.arange(mask.size(0), device=mask.device).view(-1, 1).expand_as(mask)[mask][neg_ix]
        t_ix = torch.arange(mask.size(1), device=mask.device).view(1, -1).expand_as(mask)[mask][neg_ix]
        return b_ix, t_ix

    @property
    def metric_name(self):
        return "val_loss"

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
            STEP_OUTPUT:
                if stage == "train", returns total loss.
                else returns a dictionary of metrics.
        """
        mask = self.get_mask(batch.seq_len_mask)
        x_trx = self.encoder.trx_encoder(batch)
        x_trx_masked = self.mask_x(x_trx.payload, batch.seq_len_mask, mask)

        embeddings = self.encoder.seq_encoder(PaddedBatch(x_trx_masked, batch.seq_len_mask)).payload
        out = self.head(embeddings)

        target = x_trx.payload[mask].unsqueeze(1)  # N, 1, H
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

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.shared_step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self.shared_step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self.shared_step("test", *args, **kwargs)

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