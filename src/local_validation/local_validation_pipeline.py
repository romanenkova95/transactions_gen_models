import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryAccuracy,
)

from ptls.data_load.padded_batch import PaddedBatch


class LocalValidationModel(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        hidden_size: int,
        learning_rate: float = 1e-3,
        backbone_output_type: str = "tensor",
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            backbone_embd_size (int) - size of embeddings produced by backbone model
            hidden_size (int) - hidden size for 2-layer linear prediction head
            learning_rate (float) - learning rate for prediction head training
            backbone_output_type (str) - type of output of the backbone model
                                         (e.g. torch.Tensor -- for CoLES, PaddedBatch for BestClassifier)
        """
        super().__init__()

        assert backbone_output_type in [
            "tensor",
            "padded_batch",
        ], "Unknown output type of the backbone model"

        self.backbone = backbone

        # freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pred_head = nn.Sequential(
            nn.Linear(backbone_embd_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # BCE loss for seq2seq binary classification
        self.loss = nn.BCELoss()

        self.lr = learning_rate

        self.metrics = {
            "AUROC": BinaryAUROC(),
            "PR-AUC": BinaryAveragePrecision(),
            "Accuracy": BinaryAccuracy(),
            "F1Score": BinaryF1Score(),
        }

        self.backbone_output_type = backbone_output_type

    def forward(self, inputs: PaddedBatch) -> torch.Tensor:
        """Do forward pass through the local validation model.

        Args:
            inputs (PaddedBatch) - inputs if ptls format

        Returns:
            torch.Tensor of predicted local targets
        """
        out = self.backbone(inputs)
        if self.backbone_output_type == "padded_batch":
            out = out.payload
        out = self.pred_head(out).squeeze(-1)
        return out

    def training_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Training step of the LocalValidationModel."""
        inputs, labels = batch
        preds = self.forward(inputs)

        train_loss = self.loss(preds, labels.float())
        train_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return {"loss": train_loss, "acc": train_accuracy}

    def validation_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Validation step of the LocalValidationModel."""
        inputs, labels = batch

        preds = self.forward(inputs)

        val_loss = self.loss(preds, labels.float())
        val_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return {"loss": val_loss, "acc": val_accuracy}

    def test_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Test step of the LocalValidationModel."""
        inputs, labels = batch
        preds = self.forward(inputs)

        dict_out = {"preds": preds, "labels": labels}
        for name, metric in self.metrics.items():
            metric.to(inputs.device)
            metric.update(preds, labels)

            dict_out[name] = metric.compute().item()

        return dict_out

    def on_test_epoch_end(self) -> dict[str, float]:
        """Collect test_step outputs and compute test metrics for the whole test dataset."""
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
            self.log(name, metric.compute())
        return results

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer for the LocalValidationModel."""
        opt = torch.optim.Adam(self.pred_head.parameters(), lr=self.lr)
        return opt
