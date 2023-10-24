"""Local targets validation script. """

from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames import PtlsDataModule

from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess
from src.local_validation import LocalValidationModel


def local_target_validation(
    cfg_preprop: DictConfig, cfg_validation: DictConfig
) -> pd.DataFrame:
    """Full pipeline for the sequence encoder local validation.

    Args:
        cfg_preprop (DictConfig):    Dataset config (specified in the 'config/dataset')
        cfg_validation (DictConfig): Validation config (specified in the 'config/validation')

    Returns:
        results (pd.DataFrame):      Dataframe with test metrics for each run
    """
    logger = get_logger(name=__name__)
    train, val, test = preprocess(cfg_preprop)

    logger.info("Instantiating the sequence encoder")
    # load pretrained sequence encoder
    sequence_encoder = instantiate(cfg_validation["sequence_encoder"])
    sequence_encoder.load_state_dict(torch.load(cfg_validation["path_to_state_dict"]))

    data_train = MemoryMapDataset(
        train, [SeqLenFilter(cfg_validation["val_mode"]["model"]["seq_len"])]
    )
    data_val = MemoryMapDataset(
        val, [SeqLenFilter(cfg_validation["val_mode"]["model"]["seq_len"])]
    )
    data_test = MemoryMapDataset(
        test, [SeqLenFilter(cfg_validation["val_mode"]["model"]["seq_len"])]
    )

    train_dataset: ColesDataset = instantiate(
        cfg_validation["val_mode"]["dataset"], data=data_train
    )
    val_dataset: ColesDataset = instantiate(
        cfg_validation["val_mode"]["dataset"], data=data_val
    )  ##
    test_dataset: ColesDataset = instantiate(
        cfg_validation["val_mode"]["dataset"], data=data_test
    )  ##

    datamodule: PtlsDataModule = instantiate(
        cfg_validation["datamodule"],
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
    )

    results = []
    for i in range(cfg_validation["n_runs"]):
        logger.info(
            f'Training LocalValidationModel. Run {i+1}/{cfg_validation["n_runs"]}'
        )

        seed_everything(i)

        valid_model: LocalValidationModel = instantiate(
            cfg_validation["val_mode"]["model"], backbone=sequence_encoder
        )

        val_trainer: Trainer = instantiate(
            cfg_validation["trainer"],
            callbacks=[
                # EarlyStopping(
                #     monitor="train_acc", min_delta=0.01, patience=5, mode="max"
                # ),
                LearningRateMonitor(logging_interval="step"),
            ],
        )

        val_trainer.fit(valid_model, datamodule)
        torch.save(valid_model.state_dict(), f"saved_models/validation_head_{i}.pth")

        # trainer.test() returns List[Dict] of results for each dataloader; we use a single dataloader
        metrics = val_trainer.test(valid_model, datamodule)[0]
        results.append(metrics)

    return pd.DataFrame(results)
