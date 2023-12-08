"""BinaryLocalVal class: local validation with binary targets."""

from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
)

from .local_validation_model import LocalValidationModelBase


class BinaryLocalVal(LocalValidationModelBase):
    """Module for local validation of backbone on the task of predicting a binary target."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize EventType model with pretrained backbone model and 1-layer linear prediction head.

        Args:
        ----
            backbone (nn.Module): backbone model for transactions representations
            backbone_embd_size (int): output dim of backbone
            freeze_backbone (bool): whether to freeze backbone model
            learning_rate (float): learning rate for prediction head training
        """
        pred_head = nn.Sequential(nn.Linear(backbone_embd_size, 1), nn.Flatten(0, -1))

        metrics = MetricCollection(
            {
                "AUROC": BinaryAUROC(),
                "PR-AUC": BinaryAveragePrecision(),
                "Accuracy": BinaryAccuracy(),
                "F1Score": BinaryF1Score(),
            }
        )

        # cast targets to float (long targets don't work with nn.BCEWithLogitsLoss)
        def loss(preds: Tensor, target: Tensor):
            return nn.functional.binary_cross_entropy_with_logits(preds, target.float())

        super().__init__(
            backbone=backbone,
            pred_head=pred_head,
            loss=loss,
            metrics=metrics,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            postproc=nn.Sigmoid(),
        )
