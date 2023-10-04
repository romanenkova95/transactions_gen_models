from pathlib import Path

import pandas as pd
import numpy as np

import json
import torch

import datetime

from typing import Optional

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from src.coles import CustomColesDataset, CustomColesValidationDataset, CustomCoLES
from src.local_validation import LocalValidationModel


DATASET = "churn"


@hydra.main(version_base=None, config_path="config", config_name="config_" + DATASET)
def main(cfg: DictConfig):
    cfg_model = cfg["model"]

    df = pd.read_parquet(
        Path(cfg["dataset"]["dir_path"]).joinpath(cfg["dataset"]["train_file_name"])
    )
    df["fake_local_label"] = np.ones(len(df)).astype(int)

    local_target = cfg_model["validation_dataset"]["local_target_col"]

    preprocessor = PandasDataPreprocessor(
        col_id="user_id",
        col_event_time="timestamp",
        event_time_transformation="dt_to_timestamp",  # no time preprocessing
        cols_category=["mcc_code"],
        cols_numerical=["amount", local_target],  # keep column with fake local targets
        return_records=True,
    )

    dataset = preprocessor.fit_transform(df)

    train, val_test = train_test_split(dataset, test_size=0.2, random_state=142)
    val, test = train_test_split(val_test, test_size=0.5, random_state=142)

    train_data: CustomColesDataset = instantiate(cfg_model["dataset"], data=train)
    val_data: CustomColesDataset = instantiate(cfg_model["dataset"], data=val)

    train_datamodule: PtlsDataModule = instantiate(
        cfg_model["datamodule"], train_data=train_data, valid_data=val_data
    )

    model: CustomCoLES = instantiate(cfg_model["model"])

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

    logger: TensorBoardLogger = instantiate(cfg_model["trainer_coles"]["logger"])

    trainer: Trainer = instantiate(
        cfg_model["trainer_coles"]["trainer"],
        callbacks=[model_checkpoint, early_stopping],
        logger=logger,
    )

    trainer.fit(model, train_datamodule)

    # torch.save(
    #     model.seq_encoder.state_dict(),
    #     Path(cfg_model["dir_path"]) / f"{cfg_model['save_name']}.pth",
    # )

    train_data_local: CustomColesValidationDataset = instantiate(
        cfg_model["validation_dataset"], data=train
    )
    val_data_local: CustomColesValidationDataset = instantiate(
        cfg_model["validation_dataset"], data=val
    )
    test_data_local: CustomColesValidationDataset = instantiate(
        cfg_model["validation_dataset"], data=test
    )

    val_datamodule: PtlsDataModule = instantiate(
        cfg_model["datamodule"],
        train_data=train_data_local,
        valid_data=val_data_local,
        test_data=test_data_local,
        train_batch_size=10,
        valid_batch_size=1,
        test_batch_size=1,
    )

    EMBD_SIZE = 1024
    HIDDEN_SIZE = 32

    valid_model = LocalValidationModel(
        backbone=model,
        backbone_embd_size=EMBD_SIZE,
        hidden_size=HIDDEN_SIZE,
    )

    val_trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
    )

    val_trainer.fit(valid_model, val_datamodule)

    metrics = val_trainer.test(valid_model, val_datamodule)
    print("Metrics", metrics)

    return metrics[0]["AUROC"]


if __name__ == "__main__":
    main()
