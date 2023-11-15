"""CategoricalLocalVal class: local validation with categorical targets."""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAveragePrecision,
)

from .local_validation_model import LocalValidationModelBase


class CategoricalLocalVal(LocalValidationModelBase):
    """
    Module for local validation of backbone on the task of predicting a categorical target.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        num_types: int,
        pad_value: int = 0,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize EventType model with pretrained backbone model and 1-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            backbone_embd_size (int) - output dim of backbone
            num_types (int) - number of possible event types (MCC-codes) for 'event_time' validation mode
            pad_value (int) - MCC-code corresponding to padding
            freeze_backbone (bool) - whether to freeze backbone model
            learning_rate (float) - learning rate for prediction head training
        """
        pred_head = nn.Sequential(
            nn.Linear(backbone_embd_size, num_types)
        )

        metrics = MetricCollection(
            {
                "AUROC": MulticlassAUROC(
                    num_classes=num_types,
                    ignore_index=pad_value,
                    average="weighted",
                ),
                "PR-AUC": MulticlassAveragePrecision(
                    num_classes=num_types,
                    ignore_index=pad_value,
                    average="weighted",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_types,
                    ignore_index=pad_value,
                    average="micro",
                ),
                "F1Score": MulticlassF1Score(
                    num_classes=num_types,
                    ignore_index=pad_value,
                    average="micro",
                ),
            }
        )

        super().__init__(
            backbone=backbone,
            pred_head=pred_head,
            loss=nn.CrossEntropyLoss(),
            metrics=metrics,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            postproc=nn.Softmax(1)
        )

        self.num_types = num_types
        self.pad_value = pad_value
