"""Autoencoder training script"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, open_dict
from ptls.frames import PtlsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
import torch

from src.generation.modules import VanillaAE, MLMModule
from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess
from src.local_validation.local_validation_pipeline import local_target_validation

logger = get_logger(name=__name__)


def train_autoencoder(
    cfg_preprop: DictConfig,
    cfg_dataset: DictConfig,
    cfg_model: DictConfig,
    cfg_validation: Optional[DictConfig] = None,
) -> None:
    """Train autoencoder, specified in the model config, on the data, specified by the preprocessing config.

    Args:
        cfg_preprop (DictConfig):
            the preprocessing config, for specifications see preprocess.
        cfg_dataset (DictConfig):
            the dataset config, to be partially instantiated, and later supplied the data argument,
            a list of dicts of features, outputted by preprocess.
        cfg_model (DictConfig):
            the model config, which should contain:
                - encoder: nn.Module of encoder
                - decoder: nn.Module of decoder
                - module_ae: pl.Module, used to train & validate the model, subclass of AbsAE
                - trainer_args: arguments to pass to pl.Trainer
                - datamodule_args: arguments to pass when constructing the ptls datamodule. Optional, defaults to {}
                - split_seed: randomness seed to use when splitting records. Optional, defaults to 42
        cfg_validation (DictConfig):
            the validation config, for local validation
    """
    logger.info("Preparing data:")
    train, val, test = preprocess(cfg_preprop)

    ds_factory = instantiate(cfg_dataset, _partial_=True)
    datamodule = PtlsDataModule(
        train_data=ds_factory(train),
        valid_data=ds_factory(val),
        test_data=ds_factory(test),
        **cfg_model.get("datamodule_args", {}),
    )

    # Initialize module
    from_checkpoint = cfg_model["module_ae"]["_target_"].endswith("load_from_checkpoint")
    if from_checkpoint:
        logger.info("Loading module from checkpoint...")   
        module: Union[VanillaAE, MLMModule] = call(cfg_model["module_ae"])
    else:
        logger.info("Instantiating module...")
        kwargs = {k: v for k in ["encoder", "decoder"] if (v := cfg_model.get(k))}
        
        module: Union[VanillaAE, MLMModule] = instantiate(
            cfg_model["module_ae"],
            _recursive_=False,
            **kwargs
        )

    # See if in debug mode
    fast_dev_run = cfg_model["trainer_args"].get("fast_dev_run")

    # Initialize callbacks
    ckpt_callback = ModelCheckpoint(monitor=cfg_model["checkpoint_metric"])
    lr_monitor_callback = LearningRateMonitor()
    callbacks = [ckpt_callback, lr_monitor_callback]

    if "early_stopping" in cfg_model:
        callbacks.append(instantiate(cfg_model["early_stopping"]))

    # Set up Wandb
    lightning_logger = WandbLogger(
        project="macro_micro_coles", offline=bool(fast_dev_run)
    )

    cfg_dict = {
        "preprocessing": OmegaConf.to_container(cfg_preprop),
        "dataset": OmegaConf.to_container(cfg_dataset),
        "model": OmegaConf.to_container(cfg_model),
    }

    lightning_logger.experiment.config.update(cfg_dict)

    trainer = Trainer(
        accelerator="gpu",
        logger=lightning_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        **cfg_model["trainer_args"],
    )

    trainer.fit(module, datamodule=datamodule)
    if (ckpt_path := Path(ckpt_callback.best_model_path)).is_file():
        module.load_state_dict(torch.load(ckpt_path)["state_dict"])
      
    trainer.validate(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)

    torch.save(module.encoder.state_dict(), "saved_models/coles_churn.pth")
    if cfg_validation:
        # Pass debug mode to validation
        with open_dict(cfg_validation):
            cfg_validation["trainer"]["fast_dev_run"] = fast_dev_run

        logger.info("Starting validation")
        local_validation_res = local_target_validation(
            cfg_preprop, cfg_validation
        )
        if isinstance(lightning_logger, WandbLogger):
            lightning_logger.log_table(
                dataframe=local_validation_res.describe().reset_index(), key="local_validation"
            )
        else:
            print(local_validation_res.describe())
