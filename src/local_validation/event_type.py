"""EventTypeLocalVal class: mcc_code local validation."""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from torchmetrics.classification import (
    Accuracy,
    AUROC,
    F1Score,
    AveragePrecision,
)


from ptls.data_load.padded_batch import PaddedBatch


from .local_validation_model import LocalValidationModelBase


class EventTypeLocalVal(LocalValidationModelBase):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        num_types: int,
        mcc_padd_value: int = 0,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            freeze_backbone (bool) - whether to freeze backbone model
            num_types (int) - number of possible event types (MCC-codes) for 'event_time' validation mode
            learning_rate (float) - learning rate for prediction head training
            mcc_padd_value (int) - MCC-code corresponding to padding
        """
        pred_head = nn.Sequential(
            nn.Linear(backbone_embd_size, num_types), nn.Softmax(1)
        )

        def loss(probs, target):
            return nn.functional.nll_loss(
                torch.log(probs), target, ignore_index=mcc_padd_value
            )

        metrics = MetricCollection(
            {
                "AUROC": AUROC(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                    average="weighted"
                ),
                "PR-AUC": AveragePrecision(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                    average="weighted"
                ),
                "Accuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                    average="micro"
                ),
                "F1Score": F1Score(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                    average="micro"
                ),
            }
        )

        super().__init__(
            backbone=backbone,
            pred_head=pred_head,
            loss=loss,
            metrics=metrics,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
        )

        self.num_types = num_types
        self.mcc_padd_value = mcc_padd_value

    def shared_step(
        self, batch: tuple[PaddedBatch, torch.LongTensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared step for training, valiation and testing.

        Args:
            batch (tuple[PaddedBatch, torch.Tensor]) - inputs in ptls format (PaddedBatch and labels), no sampling

        Returns a tuple of:
            * preds (torch.Tensor) - model predictions
            * target (torch.Tensor) - true target values
            * mask (torch.Tensor) - binary mask indication non-padding transactions
        """
        inputs, target = batch
        preds = self(inputs)
        return preds, target
