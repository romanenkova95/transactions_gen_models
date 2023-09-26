import pandas as pd
import numpy as np

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
from torch.utils.data.dataloader import DataLoader


class PoolingModel(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions 
        representations with pooling of information of different users.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        backbone: nn.Module,
        agregating_model: nn.Module = None,
        pooling_type: str = "mean",
        hidden_size: int = 32,
        backbone_embd_size: int = None,
        agregating_model_emb_dim: int = 0,
        learning_rate: float = 1e-3,
        backbone_output_type: str = "tensor",
        max_users_in_train_dataloader=3000,
    ) -> None:
        """Initialize method for PoolingModel

        Args:
            backbone ():  Local embeding model
            argegating_model (torch.nn.Module): Model for agregating pooled embeddings
            train_dataloader (train_dataloader from CustomColesValidationDataset): DataLoader for calculating global
                embedings from local sequences, train_batch_size need to be set equal to 1
            pooling_type (str): "max" or "mean", type of pooling
        """
        super().__init__()

        assert backbone_output_type in [
            "tensor",
            "padded_batch",
        ], "Unknown output type of the backbone model"

        self.backbone = backbone
        self.backbone_embd_size = backbone_embd_size

        # freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.agregating_model = agregating_model
        self.agregating_model_emb_dim = agregating_model_emb_dim
        if agregating_model is None:
            self.agregating_model_emb_dim = 0

        self.pooled_embegings_dataset = self.make_pooled_embegings_dataset(
            train_dataloader, pooling_type, max_users_in_train_dataloader
        )

        self.pred_head = nn.Sequential(
            nn.Linear(self.get_emb_dim(), hidden_size),
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

    def make_pooled_embegings_dataset(
        self,
        train_dataloader: DataLoader,
        pooling_type: str,
        max_users_in_train_dataloader: int,
    ) -> dict[int, torch.Tensor]:
        """Creation of pooled embeding dataset. This function for each timestamp get
        sequences in dataset which ends close to this timestamp,
        make local embedding out of them and pool them together

        Args:
            train_dataloader (train_dataloader from CustomColesValidationDataset): DataLoader for calculating global
                embedings from local sequences (train_batch_size need to be set equal to 1!)
            pooling_type (str): "max" or "mean", type of pooling

        Return:
            pooled_embegings_dataset(dict): dictionary containing timestamps and pooling vectors for that timestamps
        """
        data = {}
        for i, (x, y) in enumerate(train_dataloader):
            data[i] = {}

            embs = self.backbone(x).unsqueeze(1).detach().cpu().numpy()
            times = x.payload["event_time"][:, -1].cpu().numpy()
            for emb, time in zip(embs, times):
                data[i][time] = emb
            if i > max_users_in_train_dataloader:
                break

        data = pd.DataFrame(data).sort_index().ffill()
        pooled_embegings_dataset = {}
        for time in data.index:
            vectors = np.concatenate(data.loc[time].dropna().values)

            if pooling_type == "max":
                pooled_vector = torch.tensor(np.max(vectors, axis=0))
            elif pooling_type == "mean":
                pooled_vector = torch.tensor(np.mean(vectors, axis=0))
            else:
                raise ValueError("Unsupported pooling type.")

            pooled_embegings_dataset[time] = pooled_vector

        return pooled_embegings_dataset

    def forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Forward method."""
        batch_of_global_poolings = torch.tensor([]).to(self.device)
        for i, event_time_seq in enumerate(batch.payload["event_time"]):
            if self.agregating_model is None:
                local_pooled_emb = self.make_local_pooled_embedding(
                    event_time_seq[-1].item()
                )
                local_pooled_emb = local_pooled_emb.to(self.device)
                batch_of_global_poolings = torch.cat(
                    (batch_of_global_poolings, local_pooled_emb.unsqueeze(0))
                )
            else:
                seq_of_pooled_embs = torch.tensor([]).to(self.device)
                for time in event_time_seq:
                    local_pooled_emb = self.make_local_pooled_embedding(time.item()).to(
                        self.device
                    )
                    seq_of_pooled_embs = torch.cat(
                        (seq_of_pooled_embs, local_pooled_emb.unsqueeze(0))
                    )
                batch_of_global_poolings = torch.cat(
                    (
                        batch_of_global_poolings,
                        self.agregating_model(seq_of_pooled_embs).unsqueeze(0),
                    )
                )

        batch_of_local_embedings = self.backbone(batch)
        out = torch.cat(
            (
                batch_of_local_embedings,
                batch_of_global_poolings.to(batch_of_local_embedings.device),
            ),
            dim=1,
        )
        out = self.pred_head(out).squeeze(-1)

        return out

    def make_local_pooled_embedding(self, time: int) -> torch.Tensor:
        """Function that find the most close timestamp in self.pooled_embegings_dataset
        and return the pooling vector at this timestamp.

        Args:
            time (int): timepoint for which we are looking for pooling vector

        Return:
            pooled_vector (torch.Tensor): pooling vector for given timepoint

        """
        times = list(self.pooled_embegings_dataset.keys())
        indexes = np.where(np.array(times) < time)[0]

        if len(indexes) == 0:
            pooled_vector = torch.rand(self.backbone_embd_size)
        else:
            closest_time = times[indexes[-1]]
            pooled_vector = self.pooled_embegings_dataset[closest_time]

        if self.agregating_model is not None:
            pooled_vector = pooled_vector.to(self.device)

        return pooled_vector

    def get_emb_dim(self):
        if self.agregating_model is None:
            return 2 * self.backbone_embd_size
        else:
            return self.backbone_embd_size + self.agregating_model_emb_dim

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
