"""Main coles learning script"""
from pathlib import Path
import pickle

from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd

from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from src.coles import CustomCoLES, CustomColesDataset
from src.utils.logging_utils import get_logger


logger = get_logger(name=__name__)


def learn_coles(cfg_preprop: DictConfig, cfg_model: DictConfig) -> None:
    """Full pipeline for the coles model fitting.

    Args:
        cfg_preprop (DictConfig): Dataset config (specified in the 'config/dataset')
        cfg_model (DictConfig): Model config (specified in the 'config/model')
    """
    dataframe = pd.read_parquet(
        Path(cfg_preprop["dir_path"]).joinpath(cfg_preprop["train_file_name"])
    )
    logger.info("dataframe initialized")

    dataset_name = cfg_preprop["name"]

    path_to_preprocessor = Path(cfg_preprop["coles"]["pandas_preprocessor"]["dir_path"])
    if not path_to_preprocessor.exists():
        logger.warning("Preprocessor directory does not exist. Creating")
        path_to_preprocessor.mkdir(parents=True)
    path_to_preprocessor = path_to_preprocessor.joinpath(
        cfg_preprop["coles"]["pandas_preprocessor"]["name"]
    )

    if not path_to_preprocessor.exists():
        logger.info(
            "Preprocessor was not saved, so the fitting process will be provided"
        )
        event_time_transformation = (
            "none" if dataset_name == "age" else "dt_to_timestamp"
        )

        cols_numerical = ["amount"]
        if dataset_name != "age":
            local_target = cfg_model["validation_dataset"]["local_target_col"]
            cols_numerical += [local_target]

        preprocessor = PandasDataPreprocessor(
            col_id="user_id",
            col_event_time="timestamp",
            event_time_transformation=event_time_transformation,
            cols_category=["mcc_code"],
            cols_numerical=cols_numerical,
            cols_first_item=[
                "global_target"
            ],  # global target is duplicated, use 1st value
            return_records=True,
        )
        dataset = preprocessor.fit_transform(dataframe)
        with path_to_preprocessor.open("wb") as file:
            pickle.dump(preprocessor, file)
    else:
        with path_to_preprocessor.open("rb") as file:
            preprocessor: PandasDataPreprocessor = pickle.load(file)
        dataset = preprocessor.transform(dataframe)

    logger.info("Preparing datasets and datamodule")
    # train val splitting
    train, val = train_test_split(dataset, test_size=cfg_preprop["coles"]["test_size"])

    # Define our ColesDataset wrapper from the config
    train_data: CustomColesDataset = instantiate(cfg_model["dataset"], data=train)
    val_data: CustomColesDataset = instantiate(cfg_model["dataset"], data=val)

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
