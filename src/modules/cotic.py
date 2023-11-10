from typing import Optional
from omegaconf import DictConfig
from hydra.utils import instantiate

from ptls.frames.abs_module import ABSModule

from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class Cotic(ABSModule):
    """Cotic module in ptls format."""

    def __init__(
        self,
        encoder: DictConfig,
        head: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        optimizer_partial: DictConfig,
        lr_scheduler_partial: DictConfig,
        head_start: Optional[int] = None,
    ) -> None:
        """Initalize Cotic module.

        Args:
            seq_encoder (SeqEncoderContainer) - container with sequence continuous convolutional sequence encoder
            head (PredictionHead) - custom prediction head for Cotic model
            loss (CoticLoss) - module with Cotic losses
            metrics (CoticMetrics) - module with Cotic metrics
            optimizer_partial, lr_scheduler_partial (bool) - ptls formar arguments
            head_start (int) - if nor None, start training prediction head after this epoch.
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

        self.train_metrics = instantiate(metrics)
        self.val_metrics = self.train_metrics.copy_empty()
        self.test_metrics = self.train_metrics.copy_empty()

        self._head = instantiate(head)

        self.head_start = head_start

    def shared_step(self, batch):
        """
        batch -- PaddedBatch that is fed into CoticSeqEncoder (container: TrxEncoder + CoticEncoder)
        """
        encoded_output = self(
            batch[0]
        )  # out of CoticSeqEncoder (aka 'encoded_output' in Cotic)
        pred_times, pred_types = self._head(encoded_output.detach())

        inputs = self.seq_encoder._extract_times_and_features(
            batch[0]
        )  # format is (event_times, event_types)
        outputs = encoded_output, (
            pred_times,
            pred_types,
        )  # format is (encoded_output, (pred_times, pred_types))

        return inputs, outputs

    def training_step(self, batch, _):
        inputs, outputs = self.shared_step(batch)

        ll_loss, type_loss, time_loss = self._loss.compute_loss(
            model=self.seq_encoder.seq_encoder.feature_extractor,
            inputs=inputs,
            outputs=outputs,
        )

        self.log("train_ll_loss", ll_loss, prog_bar=True)

        if self.head_start is not None and self.current_epoch >= self.head_start:
            self.log("train_type_loss", type_loss, prog_bar=True)
            self.log("train_time_loss", time_loss, prog_bar=True)

            self.train_metrics.update(inputs, outputs)

            return {"loss": ll_loss + type_loss + time_loss}

        return {"loss": ll_loss}

    def validation_step(self, batch, _):
        inputs, outputs = self.shared_step(batch)

        ll_loss, type_loss, time_loss = self._loss.compute_loss(
            model=self.seq_encoder.seq_encoder.feature_extractor,
            inputs=inputs,
            outputs=outputs,
        )

        self.log("val_ll_loss", ll_loss, prog_bar=True)

        if self.head_start is not None and self.current_epoch >= self.head_start:
            self.log("val_type_loss", type_loss, prog_bar=True)
            self.log("val_time_loss", time_loss, prog_bar=True)

            self.val_metrics.update(inputs, outputs)

            return {"loss": ll_loss + type_loss + time_loss}

        return {"loss": ll_loss}

    def test_step(self, batch, _):
        if self.head_start is not None:
            inputs, outputs = self.shared_step(batch)

            self.test_metrics.update(inputs, outputs)

    def training_epoch_end(self, outputs):
        if self.head_start is not None and self.current_epoch >= self.head_start:
            return_time_metric, event_type_metric = self.train_metrics.compute()

            self.log(f"val_return_time_metric", return_time_metric, prog_bar=True)
            self.log(f"val_event_type_metric", event_type_metric, prog_bar=True)

    def validation_epoch_end(self, outputs):
        if self.head_start is not None and self.current_epoch >= self.head_start:
            return_time_metric, event_type_metric = self.val_metrics.compute()

            self.log(f"val_return_time_metric", return_time_metric, prog_bar=True)
            self.log(f"val_event_type_metric", event_type_metric, prog_bar=True)

    def test_epoch_end(self, outputs):
        return_time_metric, event_type_metric = self.test_metrics.compute()

        self.log(f"test_return_time_metric", return_time_metric, prog_bar=True)
        self.log(f"test_event_type_metric", event_type_metric, prog_bar=True)

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(epoch=self.current_epoch)

    @property
    def is_requires_reduced_sequence(self):
        return False

    @property
    def metric_name(self):
        return "val_ll_loss"
