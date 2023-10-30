from typing import Union
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torch import Tensor


from ptls.data_load import PaddedBatch
from ptls.frames.coles import CoLESModule

from src.generation.modules import VanillaAE


class ColesAEComboModule(VanillaAE):
    def __init__(
        self, coles_optimizer_cfg: DictConfig, ae_optimizer_cfg: DictConfig, **kwargs
    ) -> None:
        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.encoder: CoLESModule
        self.coles_optimizer_cfg = coles_optimizer_cfg
        self.ae_optimizer_cfg = ae_optimizer_cfg

        self.automatic_optimization = False

    def forward(self, x: PaddedBatch):
        output_embeddings, latent_embeddings = AbsAE.forward(
            self, x, return_latent=True
        )
        mcc_pred = self.out_mcc(output_embeddings)
        amount_pred = self.out_amount(output_embeddings)

        # zero-out padding to disable grad flow
        pad_mask = x.seq_len_mask.bool().reshape(*(amount_pred.shape))
        mcc_pred[~pad_mask] = 0
        amount_pred[~pad_mask] = 0

        return mcc_pred, amount_pred, latent_embeddings

    def _calculate_losses(
        self,
        latent_embeddings,
        target_batch,
        mcc_pred,
        amount_pred,
        mcc_target,
        amount_target,
    ):
        coles_loss: Tensor = self.encoder._loss(latent_embeddings, target_batch)  # type: ignore
        ae_loss = super()._calculate_losses(
            mcc_pred, amount_pred, mcc_target, amount_target
        )

        return {"coles": coles_loss, "ae": ae_loss}

    def shared_step(self, stage: str, batch: tuple[PaddedBatch, Tensor], *args, **kwargs):
        trx_batch, target_batch = batch
        mcc_pred, amount_pred, latent_embeddings = self(trx_batch)

        loss_dict = self._calculate_losses(
            latent_embeddings,
            target_batch,
            mcc_pred,
            amount_pred,
            trx_batch.payload["mcc_code"],
            trx_batch.payload["amount"],
        )

        metric_dict = self._calculate_metrics(
            mcc_pred,
            amount_pred,
            trx_batch.payload["mcc_code"],
            trx_batch.payload["amount"],
            trx_batch.seq_len_mask,
        )

        self.log_dict({stage: loss_dict})
        self.log_dict({stage: metric_dict})

        return {"loss": loss_dict, "metrics": metric_dict}

    def training_step(self, batch: tuple[PaddedBatch, Tensor], *args, **kwargs):
        coles_opt, ae_opt = self.optimizers()  # type: ignore

        loss_dict = self.shared_step("train", batch)["loss"]

        coles_opt.zero_grad()  # type: ignore
        self.manual_backward(loss_dict["coles"])
        coles_opt.step()

        ae_opt.zero_grad()  # type: ignore
        self.manual_backward(loss_dict["ae"]["loss"])
        ae_opt.step()

        return loss_dict
    
    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.shared_step("val", *args, **kwargs)
    
    def test_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.shared_step("test", *args, **kwargs)

    def configure_optimizers(self):
        coles_opt = self._parse_optimizer_config(
            self.coles_optimizer_cfg, self.encoder.parameters()
        )

        ae_opt = self._parse_optimizer_config(self.ae_optimizer_cfg, self.parameters())

        return coles_opt, ae_opt
