"""Main coles learning script"""
import pickle
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from ptls.frames import PtlsDataModule

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from src.coles import CustomCoLES
from src.utils.logging_utils import get_logger
from src.utils.data_utils.prepare_dataset import prepare_dataset


logger = get_logger(name=__name__)


def learn_coles(cfg_preprop: DictConfig, cfg_model: DictConfig) -> None:
    """Full pipeline for the coles model fitting.

    Args:
        cfg_preprop (DictConfig): Dataset config (specified in the 'config/dataset')
        cfg_model (DictConfig): Model config (specified in the 'config/model')
    """
    dataset = prepare_dataset(cfg_preprop, logger)

    # train val test split
    valid_size = cfg_preprop["coles"]["valid_size"]
    test_size = cfg_preprop["coles"]["test_size"]

    train, val_test = train_test_split(
        dataset,
        test_size=valid_size + test_size,
        random_state=cfg_preprop["coles"]["random_state"],
    )

    val, _ = train_test_split(
        val_test,
        test_size=test_size / (valid_size + test_size),
        random_state=cfg_preprop["coles"]["random_state"],
    )

    logger.info("Preparing datasets and datamodule")

    # Define our ColesDataset wrapper from the config
    train_data: ColesDataset = instantiate(cfg_model["dataset"], data=train)
    val_data: ColesDataset = instantiate(cfg_model["dataset"], data=val)

    # Pytorch-lifestream datamodule for the model training and evaluation
    datamodule: PtlsDataModule = instantiate(
        cfg_model["datamodule"], train_data=train_data, valid_data=val_data
    )

    # Define our CoLES wrapper from the config
    model: CustomCoLES = instantiate(cfg_model["model"])

    # Initializing and fitting the trainer for the model
    model_checkpoint: ModelCheckpoint = instantiate(
        cfg_model["trainer_coles"]["checkpoint_callback"],
        monitor=model.metric_name,
        mode="max",
    )

    callbacks = [model_checkpoint]

    if cfg_model["trainer_coles"]["enable_early_stopping"]:
        early_stopping: EarlyStopping = instantiate(
            cfg_model["trainer_coles"]["early_stopping"],
            monitor=model.metric_name,
            mode="max",
        )

        callbacks.append(early_stopping)

    coles_logger: TensorBoardLogger = instantiate(cfg_model["trainer_coles"]["logger"])

    trainer: Trainer = instantiate(
        cfg_model["trainer_coles"]["trainer"],
        callbacks=callbacks,
        logger=coles_logger,
    )

    # Training the model
    trainer.fit(model, datamodule)

    checkpoint = torch.load(model_checkpoint.best_model_path)

    model.load_state_dict(checkpoint["state_dict"])

    # Save the state_dict of the best sequence encoder
    Path("saved_models").mkdir(exist_ok=True)
    torch.save(model.get_seq_encoder_weights(), f'saved_models/{cfg_model["name"]}.pth')
