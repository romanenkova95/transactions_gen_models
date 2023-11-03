"""Main coles learning script"""
from pathlib import Path
from typing import Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from ptls.frames import PtlsDataModule

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.utilities.model_helpers import is_overridden
import wandb

from src.modules import CustomCoLES, VanillaAE
from src.utils.logging_utils import get_logger


logger = get_logger(name=__name__)


def learn(
    data: tuple[list[dict], list[dict], list[dict]], cfg: DictConfig, encoder_name: str
) -> None:
    """Full pipeline for model fitting.

    Args:
        data (tuple[list[dict], list[dict], list[dict]]):
            train, val and test sets
        cfg (DictConfig): config with the following fields:
            preprocessing (DictConfig):
                Preprocessing config to be passed to src.preprocessing.preprocess
                (specified in 'config/preprocessing')
            dataset (DictConfig):
                Dataset config, additionally passed two keyword arguments:
                 - data (list of dicts) - the data in FeatureDict format (returned by preprocess)
                 - deterministic (bool) - true if randomness needs to be off, for test/val.
                (specified in 'config/dataset')
            module (DictConfig):
                LightningModule config, with train, test and val steps (specified in 'config/module')
            datamodule (DictConfig):
                Config with
    """
    train, val, test = data

    logger.info("Preparing datasets and datamodule")
    # Define our ColesDataset wrapper from the config
    train_data: Dataset = instantiate(cfg["dataset"], data=train, deterministic=False)
    val_data: Dataset = instantiate(cfg["dataset"], data=val, deterministic=True)
    test_data: Dataset = instantiate(cfg["dataset"], data=test, deterministic=True)

    # Pytorch-lifestream datamodule for the model training and evaluation
    datamodule: PtlsDataModule = instantiate(
        cfg["datamodule"],
        train_data=train_data,
        valid_data=val_data,
        test_data=test_data,
    )

    # Instantiate the encoder (and, optionally, the decoder)
    module_args = {}
    module_args["encoder"] = cfg["encoder"]
    if "decoder" in cfg:
        module_args["decoder"] = cfg["decoder"]

    # Instantiate the LightningModule.
    # _recursive_=False to save all hyperparameters
    # as DictConfigs, to enable hp loading from lightning checkpoint
    module: Union[CustomCoLES, VanillaAE] = instantiate(
        cfg["module"], **module_args, _recursive_=False
    )

    # Set up callbacks:
    metric_mode = "min" if "loss" in module.metric_name else "max"
    model_checkpoint: ModelCheckpoint = instantiate(
        cfg["trainer"]["checkpoint_callback"],
        monitor=module.metric_name,
        mode=metric_mode,
    )

    lr_monitor = LearningRateMonitor()
    callbacks: list = [model_checkpoint, lr_monitor]

    if cfg["trainer"]["enable_early_stopping"]:
        early_stopping: EarlyStopping = instantiate(
            cfg["trainer"]["early_stopping"],
            monitor=module.metric_name,
            mode=metric_mode,
        )

        callbacks.append(early_stopping)

    trainer: Trainer = instantiate(
        cfg["trainer"]["trainer"],
        callbacks=callbacks,
    )

    if wandb.run is not None:
        wandb.config.update(OmegaConf.to_container(cfg))

    # Training the model
    seed_everything()
    trainer.fit(module, datamodule)

    # Load the checkpoint & recalculate val metrics
    if not trainer.fast_dev_run:
        checkpoint = torch.load(model_checkpoint.best_model_path)
        module.load_state_dict(checkpoint["state_dict"])
    trainer.validate(module, datamodule)

    # Optionally run test
    if is_overridden("test_step", module):
        trainer.test(module, datamodule)

    # Save the state_dict of the best sequence encoder
    saved_models_path = Path("saved_models")
    saved_models_path.mkdir(exist_ok=True)
    if not trainer.fast_dev_run:
        torch.save(
            module.encoder.state_dict(), saved_models_path / f"{encoder_name}.pth"
        )
