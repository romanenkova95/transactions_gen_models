import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from ptls.frames.abs_module import ABSModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from torchmetrics import MeanMetric

from ..nn.nhp_components import restruct_batch


class NHP(ABSModule):
    """NHP module in ptls format. Note that this module is used for both NHP and A-NHP models."""

    def __init__(
        self,
        encoder: DictConfig,
        loss: DictConfig,
        optimizer_partial: DictConfig,
        lr_scheduler_partial: DictConfig,
    ) -> None:
        """Initialize NHP / A-NHP module.

        Args:
        ----
            encoder (DictConfig): config for continuous convolutional sequence encoder instantiation
            loss (DictConfig): config for module with Cotic losses instantiation
            optimizer_partial (DictConfig): config for optimizer instantiation (ptls format)
            lr_scheduler_partial (DictConfig): config for lr scheduler instantiation (ptls format)
        """
        self.save_hyperparameters()
        enc: SeqEncoderContainer = instantiate(encoder)

        super().__init__(
            seq_encoder=enc,
            loss=instantiate(loss),
            optimizer_partial=instantiate(optimizer_partial, _partial_=True),
            lr_scheduler_partial=instantiate(lr_scheduler_partial, _partial_=True),
        )

        self.encoder = enc
        self.valid_metric = MeanMetric()

    def shared_step(self, batch: tuple[PaddedBatch, torch.Tensor]) -> torch.Tensor:
        """Shared training/validation/testing step.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)

        Retruns:
        -------
            torch.Tensor: value of the loss function
        """
        (
            time_seqs,
            time_delta,
            event_types,
            non_pad_mask,
            attention_mask,
            type_mask,
        ) = restruct_batch(
            batch[0],
            col_time=self.encoder.col_time,  # type: ignore
            col_type=self.encoder.col_type,  # type: ignore
            pad_token_id=self.encoder.seq_encoder.pad_token_id,
            num_types=self.encoder.seq_encoder.num_types,
        )

        loss = self._loss.compute_loss(
            self.encoder.seq_encoder,
            time_seqs,
            time_delta,
            event_types,
            non_pad_mask,
            attention_mask,
            type_mask,
        )

        return loss

    def training_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], _
    ) -> dict[str, torch.Tensor]:
        """Training step of the module.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)
            _: batch idx (not used here)

        Returns:
        -------
            dict with train loss
        """
        train_ll_loss = self.shared_step(batch)

        self.log("train_ll_loss", train_ll_loss, prog_bar=True)

        return {"loss": train_ll_loss}

    def validation_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], _
    ) -> dict[str, torch.Tensor]:
        """Training step of the module.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)
            _: batch idx (not used here)

        Returns:
        -------
            dict with val loss
        """
        val_ll_loss = self.shared_step(batch)
        self.valid_metric(-1 * val_ll_loss)
        self.log("val_ll_loss", val_ll_loss, prog_bar=True)

        return {"loss": val_ll_loss}

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        """Overwrite method as to fix bug in our PyTorch version."""
        scheduler.step(epoch=self.current_epoch)

    def validation_epoch_end(self, _) -> None:
        """Log loss for a validation epoch."""
        self.log("val_log_likelihood", self.valid_metric, prog_bar=True)

    @property
    def is_requires_reduced_sequence(self):
        """NHP does not reduce sequence by default."""
        return False

    @property
    def metric_name(self):
        """Validation monitoring metric name."""
        return "val_ll_loss"
