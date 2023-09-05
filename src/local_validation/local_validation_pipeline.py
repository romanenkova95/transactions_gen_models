import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import BinaryF1Score

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
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.
        
        Args:
            backbone (nn.Module) - backbone model for transactions representations
            backbone_embd_size (int) - size of embeddings produced by backbone model
            hidden_size (int) - hidden size for 2-layer linear prediction head
            learning_rate (float) - learning rate for prediction head training
        """
        super().__init__()
        
        self.backbone = backbone
        
        # freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.pred_head = nn.Sequential(
            nn.Linear(backbone_embd_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # BCE loss for seq2seq binary classification
        self.loss = nn.BCELoss()
        
        self.lr = learning_rate

    def forward(self, inputs: PaddedBatch) -> torch.Tensor:
        out = self.backbone(inputs)
        out = self.pred_head(out).squeeze(1)
        return out

    def training_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int) -> dict[str, float]:
        inputs, labels = batch
        preds = self.forward(inputs)

        train_loss = self.loss(preds, labels.float())
        train_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return {"loss": train_loss, "acc": train_accuracy}

    def validation_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int) -> dict[str, float]:
        inputs, labels = batch
        preds = self.forward(inputs)

        val_loss = self.loss(preds, labels.float())
        val_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return {"loss": val_loss, "acc": val_accuracy}
    
    def test_step(self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int) -> dict[str, float]:
        inputs, labels = batch
        preds = self.forward(inputs)

        acc_score = ((preds.squeeze() > 0.5).long() == labels).float().mean().item()
        F1Score = BinaryF1Score().to(inputs.device)
        f1_score = F1Score(preds, labels).item()

        self.log("Test acc", acc_score)
        self.log("Test f1_score", f1_score)
        
        return {"preds": preds, "labels": labels, "acc": acc_score, "f1": f1_score}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.Adam(self.pred_head.parameters(), lr=self.lr)
        return opt