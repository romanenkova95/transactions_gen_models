"""Module containtaining LocalValidationModel class."""
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryAccuracy,
    Accuracy,
    AUROC,
    F1Score,
    AveragePrecision,
)

from ptls.data_load.padded_batch import PaddedBatch

from .sampler import sliding_window_sampler
from .head_loss import LogCoshLoss


class LocalValidationModel(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        hidden_size: int,
        val_mode: str,
        num_types: Optional[int] = None,
        learning_rate: float = 1e-3,
        backbone_output_type: str = "tensor",
        backbone_embd_mode: str = "seq2vec",
        seq_len: Optional[int] = None,
        stride: Optional[int] = None,
        mask_col: str = "mcc_code",
        local_label_col: Optional[str] = None,
        mcc_padd_value: int = 0,
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            backbone_embd_size (int) - size of embeddings produced by backbone model
            hidden_size (int) - hidden size for 2-layer linear prediction head
            val_mode (str) - local validation mode (options: 'donwnstream', 'return_time' and 'event_time')
            num_types (int) - number of possible event types (MCC-codes) for 'event_time' validation mode
            learning_rate (float) - learning rate for prediction head training
            backbone_output_type (str) - type of output of the backbone model
                                         (e.g. torch.Tensor -- for CoLES, PaddedBatch for BestClassifier)
            backbone_embd_mode (str) - type of backbone embeddings:
                                       'seq2vec', if backbone transforms (bs, seq_len) -> (bs, embd_dim),
                                       'seq2seq', if backbone transforms (bs, seq_len) -> (bs, seq_len, embd_dim),
            seq_len (int) - size of the sliding widwon
            mask_col (str) - name of columns containing zero-padded values for mask creation
            local_label_col (str) - name of the columns containing local targets for 'downstream' validation mode
            mcc_padd_value (int) - MCC-code corresponding to padding
        """
        super().__init__()

        if backbone_output_type not in [
            "tensor",
            "padded_batch",
        ]:
            raise NameError(
                f"Unknown output type of the backbone model {backbone_output_type}."
            )

        if backbone_embd_mode not in [
            "seq2vec",
            "seq2seq",
        ]:
            raise NameError(f"Unknown backbone embeddings mode {backbone_embd_mode}.")

        if backbone_embd_mode == "seq2vec":
            if seq_len is None:
                raise ValueError(
                    "Specify subsequence length for sampling sliding windows."
                )

            if stride is None:
                stride = 1

            self.seq_len = seq_len
            self.stride = stride

        if val_mode not in [
            "downstream",
            "return_time",
            "event_type",
        ]:
            raise ValueError(f"Unknown validation mode {val_mode}.")

        self.backbone = backbone

        # freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        if val_mode == "downstream":
            if local_label_col is None:
                raise ValueError("Specify local_label_col for downstream validation.")

            # self.pred_head = nn.Sequential(
            #     nn.Linear(backbone_embd_size, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size, 1),
            #     nn.Sigmoid(),
            # )
            self.pred_head = nn.LSTM(backbone_embd_size, 1, batch_first=True)
            # BCE loss for seq2seq binary classification
            self.loss = nn.BCELoss()

            # metrics for binary classification
            self.metrics = {
                "AUROC": BinaryAUROC(),
                "PR-AUC": BinaryAveragePrecision(),
                "Accuracy": BinaryAccuracy(),
                "F1Score": BinaryF1Score(),
            }
            self.local_label_col = local_label_col

        elif val_mode == "return_time":
            # self.pred_head = nn.Sequential(
            #     nn.Linear(backbone_embd_size, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size, 1),
            # )
            self.pred_head = nn.LSTM(backbone_embd_size, 1, batch_first=True)
            # Custom LogCosh loss for return-time prediction
            self.loss = LogCoshLoss("mean")

            # regression metrics
            self.metrics = {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}
        else:
            if num_types is None:
                raise ValueError(
                    "Specify number of event types for next-event-type prediction."
                )

            # self.pred_head = nn.Sequential(
            #     nn.Linear(backbone_embd_size, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size, num_types),
            # )
            self.pred_head = nn.LSTM(backbone_embd_size, num_types, batch_first=True)
            # CrossEntropyLoss for next-event-type prediction
            self.loss = torch.nn.CrossEntropyLoss(
                ignore_index=mcc_padd_value, reduction="mean"
            )

            self.metrics = {
                "AUROC": AUROC(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                ),
                "PR-AUC": AveragePrecision(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                ),
                "Accuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                ),
                "F1Score": F1Score(
                    task="multiclass",
                    num_classes=num_types,
                    ignore_index=mcc_padd_value,
                ),
            }

            self.num_types = num_types

        self.lr = learning_rate

        self.backbone_output_type = backbone_output_type
        self.backbone_embd_mode = backbone_embd_mode
        self.val_mode = val_mode

        self.mask_col = mask_col
        self.mcc_padd_value = mcc_padd_value

    def _get_validation_labels(self, padded_batch: PaddedBatch) -> torch.Tensor:
        """Extract necessary target for local validation from the batch of data.

        Args:
            padded_batch (PaddedBatch) - container with zero-padded data (no sampling), shape of any feature = (batch_size, max_len)

        Returns:
            torch.Tensor containing targets
        """
        if self.val_mode == "downstream":
            # take column with prepared local targets (e.g. 'churn_target' for Churn local validation)
            target = padded_batch.payload[self.local_label_col]

        elif self.val_mode == "return_time":
            # extract event times for return time (next transaction time) prediction
            target = padded_batch.payload["event_time"]
        else:
            # extract event MCC-codes for next transaction types prediction
            target = padded_batch.payload["mcc_code"]

            # if MCC code > self.num_types than merge all these unpopular codes into 1 category
            target = torch.where(
                target >= self.num_types - 1, self.num_types - 1, target
            ).long()

        if self.backbone_embd_mode == "seq2vec":
            # crop targets, delete first seq_len transactions as there are not history windows for them
            date_len = target.shape[1]
            target = target[:, self.seq_len - 1 : date_len : self.stride]
        return target

    def _return_time_target_and_preds(
        self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Prepare targets for 'return_time' validation.

        Args:
            preds (torch.Tensor) - raw predictions (output of the pred_head)
            target (torch.Tensor) - raw targets (from the dataset with no sampling)
            mask (torch.Tensor) - raw mask indicating non-padding items

        Returns a tuple of:
            * preds (torch.Tensor) - modified predictions
            * target (torch.Tensor) - modified targets
            * mask (torch.Tensor) - modified mask
        """
        # get time differencies, do not take the last prediction, crop the mask
        target = target[:, 1:] - target[:, :-1]
        preds = preds.squeeze(-1)[:, :-1]
        mask = mask[:, 1:]
        return preds, target, mask

    def _event_type_target_and_preds(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Prepare targets for 'event_type' validation.

        Args:
            preds (torch.Tensor) - raw predictions (output of the pred_head)
            target (torch.Tensor) - raw targets (from the dataset with no sampling)

        Returns a tuple of:
            * preds (torch.Tensor) - modified predictions
            * target (torch.Tensor) - modified targets
        """
        # crop predictions and target
        target = target[:, 1:]
        # preds: (batch_size, max_len, num_types) -> (batch_size, num_types, max_len) for loss and metrics
        preds = preds[:, :-1, :].transpose(1, 2)
        return preds, target

    def forward(self, inputs: PaddedBatch) -> Tuple[torch.Tensor]:
        """Do forward pass through the local validation model.

        Args:
            inputs (PaddedBatch) - inputs if ptls format (no sampling)

        Returns a tuple of:
            * torch.Tensor of predicted targets
            * torch.Tensor with mask corresponding to non-padded times
        """
        bs = inputs.payload["event_time"].shape[0]

        if self.backbone_embd_mode == "seq2vec":
            collated_batch = sliding_window_sampler(
                inputs, seq_len=self.seq_len, stride=self.stride
            )

            out = self.backbone(collated_batch)

            embd_size = out.shape[-1]
            out = out.reshape(bs, -1, embd_size)

            # shape of mask is (batch_size, max_seq_len - seq_len), zeros correspond to windows with padding
            mask = (
                collated_batch.payload[self.mask_col]
                .reshape(bs, -1, self.seq_len)
                .ne(0)
                .all(dim=2)
            )
        else:
            # shape is (batch_size, max_seq_len, embd_dim)
            out = self.backbone(inputs)
            # shape is (batch_size, max_seq_len)
            mask = inputs.payload[self.mask_col].ne(0)

        # in case of baseline models
        if self.backbone_output_type == "padded_batch":
            out = out.payload

        preds, _ = self.pred_head(out)
        # preds = self.pred_head(out)

        return preds, mask

    def shared_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor]:
        """Shared step for training, valiation and testing.

        Args:
            batch (Tuple[PaddedBatch, torch.Tensor]) - inputs in ptls format (PaddedBatch and labels), no sampling

        Returns a tuple of:
            * preds (torch.Tensor) - model predictions
            * target (torch.Tensor) - true target values
            * mask (torch.Tensor) - binary mask indication non-padding transactions
        """
        inputs, _ = batch
        target = self._get_validation_labels(inputs)  # (B, L)
        preds, mask = self.forward(inputs)  # (B, L)

        return preds, target, mask

    def training_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Training step of the LocalValidationModel."""
        preds, target, mask = self.shared_step(batch, batch_idx)

        if self.val_mode == "downstream":
            train_loss = self.loss(preds[mask].squeeze(), target[mask].float())
            metric = ((preds[mask] > 0.5).long() == target[mask]).float().mean()
            metric_name = "acc"

        elif self.val_mode == "return_time":
            preds, target, mask = self._return_time_target_and_preds(
                preds, target, mask
            )

            train_loss = self.loss(target[mask], preds[mask])

            metric_name = "mae"
            metric = MeanAbsoluteError().to(preds.device)(
                preds[mask].squeeze(), target[mask]
            )
        else:
            preds, target = self._event_type_target_and_preds(preds, target)

            train_loss = self.loss(preds, target).mean()

            metric_name = "acc"
            metric = Accuracy(
                task="multiclass",
                num_classes=self.num_types,
                ignore_index=self.mcc_padd_value,
            ).to(preds.device)(preds, target)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_" + metric_name, metric, prog_bar=True)

        return {"loss": train_loss, metric_name: metric}

    def validation_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Validation step of the LocalValidationModel."""
        preds, target, mask = self.shared_step(batch, batch_idx)

        if self.val_mode == "downstream":
            val_loss = self.loss(preds[mask].squeeze(), target[mask].float())
            metric = ((preds[mask] > 0.5).long() == target[mask]).float().mean()
            metric_name = "acc"

        elif self.val_mode == "return_time":
            preds, target, mask = self._return_time_target_and_preds(
                preds, target, mask
            )

            val_loss = self.loss(target[mask], preds[mask])

            metric_name = "mae"
            metric = MeanAbsoluteError().to(preds.device)(
                preds[mask].squeeze(), target[mask]
            )
        else:
            preds, target = self._event_type_target_and_preds(preds, target)

            val_loss = self.loss(preds, target).mean()

            metric_name = "acc"
            metric = Accuracy(
                task="multiclass",
                num_classes=self.num_types,
                ignore_index=self.mcc_padd_value,
            ).to(preds.device)(preds, target)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_" + metric_name, metric, prog_bar=True)

        return {"loss": val_loss, metric_name: metric}

    def test_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Test step of the LocalValidationModel."""
        preds, target, mask = self.shared_step(batch, batch_idx)

        if self.val_mode == "downstream":
            preds = preds[mask].squeeze()
            target = target[mask]
        elif self.val_mode == "return_time":
            preds, target, mask = self._return_time_target_and_preds(
                preds, target, mask
            )
            preds = preds[mask]
            target = target[mask]
        else:
            preds, target = self._event_type_target_and_preds(preds, target)

        dict_out = {"preds": preds, "labels": target}

        for name, metric in self.metrics.items():
            metric.to(preds.device)
            metric.update(preds, target)
            dict_out[name] = metric.compute().item()

        return dict_out

    def on_test_epoch_end(self) -> Dict[str, float]:
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
