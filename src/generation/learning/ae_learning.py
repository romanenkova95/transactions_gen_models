"""Autoencoder training script"""

from typing import Optional
import pandas as pd
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from ptls.frames import PtlsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger

from src.generation.modules.base import AbsAE
from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess
from src.local_validation.local_validation_pipeline import local_target_validation

logger = get_logger(name=__name__)


def train_autoencoder(
    cfg_preprop: DictConfig, 
    cfg_dataset: DictConfig, 
    cfg_model: DictConfig, 
    cfg_validation: Optional[DictConfig] = None
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
    train, val, test = preprocess(cfg_preprop)

    ds_factory = instantiate(cfg_dataset, _partial_=True)
    datamodule = PtlsDataModule(
        train_data=ds_factory(train),
        valid_data=ds_factory(val),
        test_data=ds_factory(test),
        **cfg_model.get("datamodule_args", {}),
    )

    if cfg_model["module_ae"]["_target_"].endswith("load_from_checkpoint"):
        logger.info("Loading module from checkpoint...")
        module: AbsAE = call(cfg_model["module_ae"])
    else:
        logger.info("Instantiating module...")
        module: AbsAE = instantiate(cfg_model["module_ae"], _recursive_=False)(
            encoder_config=cfg_model["encoder"], decoder_config=cfg_model["decoder"]
        )

    callbacks = []

    if "fast_dev_run" not in cfg_model["trainer_args"]:
        lightning_logger = WandbLogger(project="macro_micro_coles")

        cfg_dict = {
            "preprocessing": OmegaConf.to_container(cfg_preprop),
            "dataset": OmegaConf.to_container(cfg_dataset),
            "model": OmegaConf.to_container(cfg_model)
        }

        lightning_logger.experiment.config.update(cfg_dict)
        callbacks.append(LearningRateMonitor())
    else:
        lightning_logger = DummyLogger()
        
    if "early_stopping" in cfg_model:
        callbacks.append(
            instantiate(cfg_model["early_stopping"])
        )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=lightning_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        **cfg_model["trainer_args"],
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)[0]
        
    if cfg_validation and "fast_dev_run" not in cfg_model["trainer_args"]:
        local_validation_res = local_target_validation(
            cfg_preprop, 
            cfg_validation,
            (train, val, test), 
            module.encoder
        )
        if isinstance(lightning_logger, WandbLogger):
            lightning_logger.log_table(
                dataframe=local_validation_res.describe(), 
                key="local_validation"
            )
        else:
            print(local_validation_res)
