from typing import Any, Optional, Union
from omegaconf import DictConfig
from hydra.utils import instantiate
from ptls.data_load import PaddedBatch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion
import torch
from torch import Tensor
from torchmetrics import AUROC, AveragePrecision, F1Score, MeanMetric, MetricCollection

from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn import TrxEncoder
from ptls.nn import PBL2Norm
from ptls.data_load.padded_batch import PaddedBatch

from .vanilla import VanillaAE

class MLMModule(VanillaAE):
    """Masked Language Model (MLM) from [ROBERTA](https://arxiv.org/abs/1907.11692).
    Out of replace_proba tokens:
     - Mask 80% with token=num_tokens - 1;
     - replace 10% with random tokens;
     - keep 10% as-is.
    Loss is reconstruction loss on the replace_proba tokens

    Parameters
    ----------
    replace_proba (float):
        Fraction of tokens to calculate loss on
    *args, **kwargs:
        passed to VanillaAE
    """

    def __init__(
        self,
        replace_proba: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.replace_proba = replace_proba
        self.mask_token = self.encoder.trx_encoder.embeddings["mcc_code"].num_embeddings - 1 # type: ignore
    
    def forward(self, batch: PaddedBatch):
        """Mask the mcc-codes of given batch & pass them through encoder and decoder

        Args:
            batch (PaddedBatch): input batch

        Returns:
            Tensor, Tensor, Union[Tensor, PaddedBatch], Tensor: 
                Same as VanillaAE
        """
        nonpad_mask = batch.seq_len_mask.bool()
        aug_mask = torch.bernoulli(nonpad_mask.float() * self.replace_proba).bool()
        
        mcc_codes_new = batch.payload["mcc_code"].clone()
        mcc_codes_new[aug_mask] = self.get_aug_tokens(mcc_codes_new[aug_mask])
        
        amount_new = batch.payload["amount"].clone()
        amount_new[aug_mask] = 0

        batch_new = PaddedBatch({
            "mcc_code": mcc_codes_new,
            "amount": amount_new
        }, batch.seq_lens)
        
        mcc_preds, amount_preds, latent_embs, _ = super().forward(batch_new)        
        return mcc_preds, amount_preds, latent_embs, aug_mask

    def get_aug_tokens(self, aug_tokens):
        shuffled_tokens = aug_tokens[torch.randperm(aug_tokens.shape[0])]
        rand = torch.rand_like(aug_tokens, dtype=torch.float32)
        
        return torch.where(
            rand < 0.8,
            self.mask_token,
            torch.where(
                rand < 0.9,
                shuffled_tokens,
                aug_tokens
            )
        )
        
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = super().configure_optimizers() # type: ignore
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.optimizer_dictconfig["lr"],
            self.trainer.max_steps or self.trainer.estimated_stepping_batches # type: ignore
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
