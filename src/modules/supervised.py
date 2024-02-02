import logging
import torch
import torchmetrics

from hydra.utils import instantiate

from ptls.frames.supervised.seq_to_target import SequenceToTarget
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from ptls.nn import Head

logger = logging.getLogger(__name__)


class CustomSeqToTarget(SequenceToTarget):
    def __init__(
        self,
        encoder: torch.nn.Module,
        num_classes: int,
        loss: torch.nn.Module,
        metric_list: torchmetrics.Metric,
        optimizer_partial,
        lr_scheduler_partial,
        **kwargs
    ):
        super().__init__(
            seq_encoder=instantiate(encoder),
            loss=instantiate(loss),
            metric_list=instantiate(
                metric_list,
                _convert_="all",
            ),
            optimizer_partial=instantiate(optimizer_partial, _convert_="partial"),
            lr_scheduler_partial=instantiate(lr_scheduler_partial, _convert_="partial"),
            **kwargs
        )
        self.head = Head(
            self.seq_encoder.embedding_size,
            objective="classification",
            num_classes=num_classes,
        )

    @property
    def encoder(self):
        return self.seq_encoder

    @property
    def metric_name(self):
        """The name of the metric to monitor."""
        return "val_loss"

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric
    ) -> None:
        """Return the super method just for lightning to think it's overriden."""
        return super().lr_scheduler_step(scheduler, optimizer_idx, metric)

    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.test_metrics.items():
            mf(y_h, y)

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        self.log("val_loss", self.loss(y_h, y))
        for name, mf in self.valid_metrics.items():
            mf(y_h, y)

    def configure_optimizers(self):
        if self.hparams.pretrained_lr is not None:
            if self.hparams.pretrained_lr == "freeze":
                for p in self.seq_encoder.parameters():
                    p.requires_grad = False
                logger.info("Created optimizer with frozen encoder")
                parameters = self.parameters()
            else:
                parameters = [
                    {
                        "params": self.seq_encoder.parameters(),
                        "lr": self.hparams.pretrained_lr,
                    },
                    {
                        "params": self.head.parameters()
                    },  # use predefined lr from `self.optimizer_partial`
                ]
                logger.info("Created optimizer with two lr groups")
        else:
            parameters = self.parameters()

        optimizer = self.optimizer_partial(parameters)
        scheduler = self.lr_scheduler_partial(optimizer)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": self.metric_name,
        }
