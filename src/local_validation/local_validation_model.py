"""Module containtaining LocalValidationModel class."""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Metric, MetricCollection

from ptls.data_load.padded_batch import PaddedBatch


class LocalValidationModelBase(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        pred_head: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: MetricCollection,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        postproc: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations.
            pred_head (nn.Module) - prediction head for target prediction.
            loss (Callable) - the loss to optimize while training. Called with (preds, targets).
            metrics (MetricCollection) - collection of metrics to track in train, val, test steps.
            freeze_backbone (bool) - whether to freeze backbone weights while training.
            learning_rate (float) - learning rate for prediction head training.
            postproc (Callable) - postprocessing function to apply to predictions before metric calculation.
        """
        super().__init__()
        self.backbone = backbone

        # freeze backbone model
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.freeze_backbone = freeze_backbone

        self.lr = learning_rate
        self.pred_head = pred_head
        self.loss = loss
        self.train_metrics = metrics.clone("Train")
        self.val_metrics = metrics.clone("Val")
        self.test_metrics = metrics.clone("Test")
        self.metric_name = "val_loss"
        self.postproc = postproc or nn.Identity()
        
    def train(self, mode: bool = True): # override train to disable training when frozen
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            
        return self

    def forward(self, inputs: PaddedBatch) -> tuple[torch.Tensor]:
        """Do forward pass through the local validation model.

        Args:
            inputs (PaddedBatch) - inputs if ptls format (no sampling)

        Returns a tuple of:
            * torch.Tensor of predicted targets
            * torch.Tensor with mask corresponding to non-padded times
        """
        out = self.backbone(inputs)
        preds = self.pred_head(out)
        return preds

    def shared_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized shared_step for model-learning with targets. Overload if neccessary

        Args:
            batch (tuple[PaddedBatch, torch.Tensor]):
                Tuple of PaddedBatch (passed to forward) & targets
            batch_idx (int): ignored

        Returns:
            tuple[torch.Tensor, torch.Tensor]: preds, target
        """
        inputs, target = batch
        preds = self(inputs)
        return preds, target

    def training_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int):
        """Training step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)
        train_loss = self.loss(preds, target)
        self.train_metrics(self.postproc(preds), target)

        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)  # type: ignore

        return train_loss

    def validation_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int):
        """Validation step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)
        val_loss = self.loss(preds, target)
        self.val_metrics(self.postproc(preds), target)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)  # type: ignore

    def test_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int):
        """Test step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)

        self.test_metrics(self.postproc(preds), target)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer for the LocalValidationModel."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)