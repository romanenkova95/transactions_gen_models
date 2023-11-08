"""EventTimeLocalVal class: time difference local validation."""
from typing import Literal
import torch.nn as nn
from torchmetrics import MeanSquaredError, MetricCollection, R2Score
from .local_validation_model import LocalValidationModelBase
from .head_loss import LogCoshLoss


class NumericLocalVal(LocalValidationModelBase):
    """
    PytorchLightningModule for local validation of backbone model (e.g. CoLES)
    on the task of predicting a numeric target.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        loss_type: Literal["mse", "logcosh"] = "logcosh"
    ) -> None:
        """Initialize model with pretrained backbone model and 1-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            backbone_embd_size (int) - output dim of backbone
            freeze_backbone (bool) - whether to freeze backbone model
            learning_rate (float) - learning rate for prediction head training
        """
        pred_head = nn.Sequential(
            nn.Linear(backbone_embd_size, 1),
            nn.Flatten(0, -1)
        )

        metrics = MetricCollection(
            {
                "R2": R2Score(),
                "MSE": MeanSquaredError(),
            }
        )
        
        loss_dict = {
            "mse": nn.MSELoss(),
            "logcosh": LogCoshLoss()
        }

        super().__init__(
            backbone=backbone,
            pred_head=pred_head,
            loss=loss_dict[loss_type],
            metrics=metrics,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
        )
