from typing import Literal, Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn, Tensor
import numpy as np

from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, r2_score

from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from src.generation.decoders.base import AbsDecoder
from src.generation.modules.base import AbsAE
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class VanillaAE(AbsAE):
    """A vanilla autoencoder, without masking, just encodes original sequence and then restores it.
    Logs train/val/test losses, which are comprised of:
     - a cross-entropy loss for mcc codes,
     - a mean-squared-error loss for amounts,
    and train/val/test metrics:
     - a macro-averaged multiclass f1-score on mcc codes
     - a macro-averaged multiclass auroc score on mcc codes
     - an r2-score on amounts

     Attributes:
        out_amount (nn.Linear):
            A linear layer, which restores the transaction amounts.
        out_mcc (nn.Linear):
            A linear layer, which restores the transaction mcc codes.
        amount_loss_weight (float):
            Normalized loss weight for the transaction amount MSE loss.
        mcc_loss_weight (float):
            Normalized loss weight for the transaction mcc code CE loss.

    Notes:
        amount_loss_weight, mcc_loss_weight are normalized so that amount_loss_weight + mcc_loss_weight = 1.
        This is done to remove one hyperparameter. Loss gradient size can be managed separately through lr.

    """

    def __init__(
        self,
        loss_weights: dict[Literal["amount", "mcc"], float],
        optimizer_config: DictConfig,
        **kwargs,
    ) -> None:
        """Initialize VanillaAE internal state.

        Args:
            loss_weights (dict):
                A dictionary with keys "amount" and "mcc", mapping them to the corresponding loss weights
            optimizer_config (DictConfig):
                A dict config with an optimizer key and optionally an lr_scheduler key.
                Both optimizer & scheduler are partially instantiated, and then initialized with
                model parameters & optimizer respectfully.
                lr_scheduler may be either a torch lr_scheduler instance,
                or a dict with lr_scheduler config (see configure_optimizers docs).
        """
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.out_amount = nn.Linear(self.ae_output_size, 1)
        self.out_mcc = nn.Linear(self.ae_output_size, self.mcc_vocab_size + 1)

        self.lr = optimizer_config["optimizer"]["lr"]
        self.optimizer_config = optimizer_config
        self.amount_loss_weight = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weight = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

    def forward(
        self,
        batch: PaddedBatch,
    ) -> tuple[Tensor, Tensor]:
        """Run the forward pass of the VanillaAE module.
        Pass the batch through BaseAE.forward, and afterwards pass it through out_mcc & out_amount
        to get the respective targets.

        Args:
            batch (PaddedBatch): Input batch of raw transactional data.

        Returns:
            tuple[Tensor, Tensor]:
                tuple of tensors:
                    - Predicted mcc logits, shape (B, L, mcc_vocab_size + 1)
                    - predicted amounts, shape (B, L)

        Notes:
            The padding elements, determined by the padding mask of the input PaddedBatch,
            are zeroed out to prevent gradient flow.

        """
        seqs_after_lstm = super().forward(batch)  # supposedly (B * S, L, E)

        mcc_rec: Tensor = self.out_mcc(seqs_after_lstm)
        amount_rec: Tensor = self.out_amount(seqs_after_lstm).squeeze(dim=-1)

        # zero-out padding to disable grad flow
        pad_mask = batch.seq_len_mask.bool().reshape(*(amount_rec.shape))
        mcc_rec[~pad_mask] = 0
        amount_rec[~pad_mask] = 0

        # squeeze for amount is required to reduce last dimension
        return (mcc_rec, amount_rec)

    def _calculate_metrics(
        self,
        mcc_preds: Tensor,
        amt_value: Tensor,
        mcc_orig: Tensor,
        amt_orig: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the metrics

        Args:
            mcc_preds (Tensor): predicted mcc logits, (B, L, mcc_vocab_size)
            amt_value (Tensor): predicted amounts, (B, L)
            mcc_orig (Tensor): original mccs, (B, L)
            amt_orig (Tensor): original amounts, (B, L)
            mask (Tensor): mask of non-padding elements

        Returns:
            dict[str, Tensor]: Dictionary of metrics, with keys mcc_auroc, mcc_f1, amt_r2
        """
        with torch.no_grad():
            mcc_orig = mcc_orig[mask]
            mcc_preds = mcc_preds[mask].reshape((*mcc_orig.shape, -1))

            labels = mcc_orig.unique()
            num_classes = len(labels)
            mcc_orig = torch.argwhere(mcc_orig[:, None] == labels[None, :])[:, 1]
            mcc_preds = mcc_preds[:, labels]

            return {
                "mcc_auroc": multiclass_auroc(
                    mcc_preds, mcc_orig, average="macro", num_classes=num_classes
                ).cpu(),
                "mcc_f1": multiclass_f1_score(
                    mcc_preds, mcc_orig, average="macro", num_classes=num_classes
                ).cpu(),
                "amt_r2": r2_score(
                    amt_value[mask],
                    amt_orig[mask],
                ).cpu(),
            }

    def _calculate_losses(
        self,
        mcc_rec: Tensor,
        amount_rec: Tensor,
        mcc_orig: Tensor,
        amount_orig: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the losses, weigh them with respective weights

        Args:
            mcc_rec (Tensor): Predicted mcc logits, (B, L, mcc_vocab_size).
            amount_rec (Tensor): Predicted amounts, (B, L).
            mcc_orig (Tensor): Original mcc codes.
            amount_orig (Tensor): Original amounts.

        Returns:
            Dictionary of losses, with keys loss, loss_mcc, loss_amt.
        """
        # Lengths tensor

        mcc_loss = self.mcc_criterion(mcc_rec.transpose(2, 1), mcc_orig)
        amount_loss = torch.log(self.amount_criterion(amount_rec, amount_orig))

        total_loss = (
            self.mcc_loss_weight * mcc_loss + self.amount_loss_weight * amount_loss
        )

        return {"loss": total_loss, "loss_mcc": mcc_loss, "loss_amt": amount_loss}

    def _all_forward_step(self, batch: PaddedBatch):
        """Run the forward step, calculate the losses and the metrics

        Args:
            batch (PaddedBatch): Input

        Returns:
            tuple[dict, dict]: Dictionary of losses, dictionary of metrics.
        """
        mcc_rec, amount_rec = self(batch)  # (B * S, L, MCC_N), (B * S, L)
        mcc_orig = batch.payload["mcc_code"]
        amount_orig = batch.payload["amount"]

        loss_dict = self._calculate_losses(mcc_rec, amount_rec, mcc_orig, amount_orig)

        metric_dict = self._calculate_metrics(
            mcc_rec, amount_rec, mcc_orig, amount_orig, batch.seq_len_mask.bool()
        )

        return loss_dict, metric_dict

    def _step(
        self,
        stage: str,
        batch: PaddedBatch,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Generalized function to do a train/val/test step.

        Args:
            stage (str): train, val, or test, depending on the stage.
            batch (PaddedBatch): Input.
            batch_idx (int): ignored

        Returns:
            STEP_OUTPUT:
                if stage == "train", returns total loss.
                else returns a dictionary of metrics.
        """
        loss_dict, metric_dict = self._all_forward_step(batch)

        self.log_dict(
            {f"{stage}_{k}": v for k, v in loss_dict.items()},
            on_step=(stage == "train"),
            on_epoch=(stage != "train"),
            batch_size=batch.seq_feature_shape[0],
        )

        self.log_dict(
            {f"{stage}_{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.seq_feature_shape[0],
        )

        if stage == "train":
            return loss_dict["loss"]
        else:
            return metric_dict

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("test", *args, **kwargs)

    def predict_step(
        self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run the predict step: forward pass for the input batch, and trim padding in output.

        Args:
            batch (PaddedBatch): input padded batch
            batch_idx (int): ignored
            dataloader_idx (int, optional): ignored

        Returns:
            tuple[list[Tensor], list[Tensor]]:
                - list of predicted mcc logits, (B, L_i, mcc_vocab_size)
                - list of predicted amounts, (B, L_i)
                Note that L_i (i=0...B-1) is different for each element of the batch,
                for this reason we return a list and not a tensor.
        """
        mcc_rec: Tensor  # (B, L, MCC_NUM)
        amount_rec: Tensor  # (B, L)
        mcc_rec, amount_rec = self(batch)
        lens_mask = batch.seq_len_mask.bool()
        lens = batch.seq_lens

        mcc_rec_trim = mcc_rec[lens_mask]
        amount_rec_trim = amount_rec[lens_mask]

        return mcc_rec_trim.split(lens), amount_rec_trim.split(lens)

    def configure_optimizers(self):
        return self._parse_optimizer_config(self.optimizer_config, self.parameters())
