"""Main coles learning script"""
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from ptls.frames import PtlsDataModule
from ptls.frames.coles import ColesDataset

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.coles import CustomCoLES, CustomColesDataset
from src.preprocessing import preprocess
from src.utils.logging_utils import get_logger


logger = get_logger(name=__name__)


def learn_coles(
    cfg_preprop: DictConfig, cfg_dataset: DictConfig, cfg_model: DictConfig
) -> None:
    """Full pipeline for the coles model fitting.

    Args:
        cfg_preprop (DictConfig): Preprocessing config (specified in 'config/preprocessing')
        cfg_dataset (DictConfig): Dataset config (specified in 'config/dataset')
        cfg_model (DictConfig): Model config (specified in 'config/model')
    """
    train, val, test = preprocess(cfg_preprop)

    logger.info("Preparing datasets and datamodule")
    # Define our ColesDataset wrapper from the config
    train_data: CustomColesDataset = instantiate(cfg_dataset, data=train)
    val_data: CustomColesDataset = instantiate(cfg_dataset, data=val)

    # Pytorch-lifestream datamodule for the model training and evaluation
    datamodule: PtlsDataModule = instantiate(
        cfg_model["datamodule"], train_data=train_data, valid_data=train_data  #
    )

    # Define our CoLES wrapper from the config
    model: CustomCoLES = instantiate(cfg_model["model"])

    # Initializing and fitting the trainer for the model
    model_checkpoint: ModelCheckpoint = instantiate(
        cfg_model["trainer_coles"]["checkpoint_callback"],
        monitor=model.metric_name,
        mode="max",
    )

    callbacks: list = [model_checkpoint]

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
    torch.save(
        model.get_seq_encoder_weights(),
        f'saved_models/{cfg_model["name"]}_{cfg_model["dataset"]["splitter"]["min_len"]}_{cfg_model["dataset"]["splitter"]["max_len"]}.pth',
    )
