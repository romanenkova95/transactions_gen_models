from typing import Any, Dict, Optional, Tuple, Union
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
    def __init__(
        self,
        loss_weights: Dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.out_amount = nn.Linear(self.ae_output_size, 1)
        self.out_mcc = nn.Linear(self.ae_output_size, self.mcc_vocab_size + 1)

        self.amount_loss_weights = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weights = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

    def forward(
        self,
        batch: PaddedBatch,
    ) -> tuple[Tensor, Tensor]:
        seqs_after_lstm = super().forward(batch)  # supposedly (B * S, L, E)

        mcc_rec = self.out_mcc(seqs_after_lstm)
        amount_rec = self.out_amount(seqs_after_lstm)

        # zero-out padding to disable grad flow
        mcc_rec[~batch.seq_len_mask.bool()] = 0
        amount_rec[~batch.seq_len_mask.bool()] = 0

        # squeeze for amount is required to reduce last dimension
        return (mcc_rec, amount_rec.squeeze(dim=-1))

    def _calculate_metrics(
        self,
        mcc_preds: Tensor,
        amt_value: Tensor,
        mcc_orig: Tensor,
        amt_orig: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            mcc_orig = mcc_orig[mask]
            mcc_preds = mcc_preds[mask].reshape((*mcc_orig.shape, -1))

            labels = mcc_orig.unique()
            num_classes = len(labels)
            mcc_orig = torch.argwhere(mcc_orig[:, None] == labels[None, :])[:, 1]
            mcc_preds = mcc_preds[:, labels]
            
            return (
                multiclass_auroc(
                    mcc_preds,
                    mcc_orig,
                    average="macro",
                    num_classes=num_classes
                ).cpu(),
                multiclass_f1_score(
                    mcc_preds,
                    mcc_orig,
                    average="macro",
                    num_classes=num_classes
                ).cpu(),
                r2_score(
                    amt_value[mask],
                    amt_orig[mask],
                ).cpu(),
            )

    def _calculate_losses(
        self,
        mcc_rec: Tensor,
        amount_rec: Tensor,
        mcc_orig: Tensor,
        amount_orig: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Lengths tensor

        mcc_loss = self.mcc_criterion(mcc_rec.transpose(2, 1), mcc_orig)
        amount_loss = torch.log(self.amount_criterion(amount_rec, amount_orig))

        total_loss = (
            self.mcc_loss_weights * mcc_loss + self.amount_loss_weights * amount_loss
        )

        return (total_loss, (mcc_loss, amount_loss))

    def _all_forward_step(self, batch: PaddedBatch):
        mcc_rec, amount_rec = self(batch)  # (B * S, L, MCC_N), (B * S, L)
        mcc_orig = batch.payload["mcc_code"]
        amount_orig = batch.payload["amount"]

        total_loss, (mcc_loss, amount_loss) = self._calculate_losses(
            mcc_rec, amount_rec, mcc_orig, amount_orig
        )

        auroc_mcc, f1_mcc, r2_amount = self._calculate_metrics(
            mcc_rec, amount_rec, mcc_orig, amount_orig, batch.seq_len_mask.bool()
        )

        return (total_loss, (mcc_loss, amount_loss), (auroc_mcc, f1_mcc, r2_amount))

    def _step(
        self,
        stage: str,
        batch: PaddedBatch,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        if not self.trainer:
            raise ValueError("No trainer!")

        (
            loss,
            (mcc_loss, amount_loss),
            (auroc_mcc, f1_mcc, r2_amount),
        ) = self._all_forward_step(batch)

        log_loss_params: Dict[str, Any] = dict(
            on_step=(stage == "train"), on_epoch=(stage != "train"), batch_size=batch.seq_feature_shape[0]
        )
        
        log_metric_params: Dict[str, Any] = dict(
            on_step=False, on_epoch=True, batch_size=batch.seq_feature_shape[0]
        )

        self.log(f"{stage}_loss", loss.detach().cpu(), prog_bar=True, **log_loss_params)
        self.log(f"{stage}_loss_mcc", mcc_loss.detach().cpu(), **log_loss_params)
        self.log(f"{stage}_loss_amt", amount_loss.detach().cpu(), **log_loss_params)
        
        self.log(f"{stage}_mcc_auroc", auroc_mcc, **log_metric_params)
        self.log(f"{stage}_mcc_f1", f1_mcc, **log_metric_params)
        self.log(f"{stage}_amt_r2", r2_amount, **log_metric_params)

        return loss

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("test", *args, **kwargs)
    
    def predict_step(self, batch: PaddedBatch, batch_idx: int, dataloader_idx: int = 0) -> Any:
        mcc_rec: Tensor # (B, L, MCC_NUM)
        amount_rec: Tensor # (B, L)
        mcc_rec, amount_rec = self(batch)
        lens_mask = batch.seq_len_mask.bool()
        lens = batch.seq_lens
    
        mcc_rec_trim = mcc_rec[lens_mask]
        amount_rec_trim = amount_rec[lens_mask]
        
        return mcc_rec_trim.split(lens), amount_rec_trim.split(lens)
