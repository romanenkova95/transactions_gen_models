"""Main coles learning script"""
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from src.coles import CustomCoLES, CustomColesDataset
from src.preprocessing import preprocess
from src.utils.logging_utils import get_logger


logger = get_logger(name=__name__)


def learn_coles(cfg_preprop: DictConfig, cfg_dataset: DictConfig, cfg_model: DictConfig) -> None:
    """Full pipeline for the coles model fitting.

    Args:
        cfg_preprop (DictConfig): Preprocessing config (specified in 'config/preprocessing')
        cfg_dataset (DictConfig): Dataset config (specified in 'config/dataset')
        cfg_model (DictConfig): Model config (specified in 'config/model')
    """
    dataset = preprocess(cfg_preprop)
    logger.info("Preparing datasets and datamodule")
    # train val splitting
    train, val = train_test_split(dataset, test_size=cfg_model["test_size"])

    # Define our ColesDataset wrapper from the config
    train_data: CustomColesDataset = instantiate(cfg_dataset, data=train)
    val_data: CustomColesDataset = instantiate(cfg_dataset, data=val)

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

    early_stopping: EarlyStopping = instantiate(
        cfg_model["trainer_coles"]["early_stopping"],
        monitor=model.metric_name,
        mode="max",
    )

    coles_logger: TensorBoardLogger = instantiate(cfg_model["trainer_coles"]["logger"])

    trainer: Trainer = instantiate(
        cfg_model["trainer_coles"]["trainer"],
        callbacks=[model_checkpoint, early_stopping],
        logger=coles_logger,
    )

    trainer.fit(model, datamodule)
