"""Local targets validation script."""

import warnings
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from ptls.frames import PtlsDataModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger, WandbLogger

from src.utils.create_trainer import create_trainer
from src.utils.logging_utils import get_logger

from .local_validation_model import LocalValidationModelBase


def local_target_validation(
    data: tuple[list[dict], list[dict], list[dict]],
    cfg_encoder: DictConfig,
    cfg_validation: DictConfig,
    cfg_logger: DictConfig,
    encoder_name: str,
    val_name: str,
    is_deterministic: Optional[bool] = None,
) -> dict[str, float]:
    """Full pipeline for the sequence encoder local validation.

    Args:
    ----
        data (tuple[list[dict], list[dict], list[dict]]):
            train, val & test sets
        cfg_encoder (DictConfig):
            Encoder config (specified in 'config/encoder')
        cfg_validation (DictConfig):
            Validation config (specified in 'config/validation')
        cfg_logger (DictConfig):
            The config to use when creating the logger.
        encoder_name (str):
            Name of used encoder (for logging & saving)
        val_name (str):
            Name of validation (for logging & saving)
        is_deterministic (bool):
            Flag which allows you to override the default dataset creation behaviour.
            If True, train & val & test are all deterministic. 
            If False, all of them are shuffled.
            If None, make train nondeterministic and val/test deterministic.

    Returns:
    -------
        results (dict[str]):
            Metrics on test set.
    """
    logger = get_logger(name=__name__)
    train, val, test = data

    logger.info("Instantiating the sequence encoder")
    # load pretrained sequence encoder
    sequence_encoder: SeqEncoderContainer = instantiate(
        cfg_encoder, is_reduce_sequence=cfg_validation["is_reduce_sequence"]
    )
    encoder_state_dict_path = Path(f"saved_models/{encoder_name}.pth")
    if encoder_state_dict_path.exists():
        sequence_encoder.load_state_dict(torch.load(encoder_state_dict_path))
    else:
        warnings.warn(
            "No encoder state dict found! Validating without pre-loading state-dict..."
        )

    train_deterministic = False if is_deterministic is None else is_deterministic
    val_deterministic = True if is_deterministic is None else is_deterministic

    train_dataset = call(
        cfg_validation["dataset"], data=train, deterministic=train_deterministic
    )
    val_dataset = call(
        cfg_validation["dataset"], data=val, deterministic=val_deterministic
    )
    test_dataset = call(
        cfg_validation["dataset"], data=test, deterministic=val_deterministic
    )

    datamodule: PtlsDataModule = instantiate(
        cfg_validation["datamodule"],
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
    )

    logger.info("Training LocalValidationModel")

    valid_model: LocalValidationModelBase = instantiate(
        cfg_validation["module"], backbone=sequence_encoder
    )

    val_trainer: Trainer = create_trainer(
        logger=cfg_logger,
        metric_name=valid_model.metric_name,
        **cfg_validation["trainer"],
    )

    # Make wandb/comet log runs to different metric graphs
    if isinstance(val_trainer.logger, (WandbLogger, CometLogger)):
        val_trainer.logger._prefix = val_name

    val_trainer.fit(valid_model, datamodule)
    if not val_trainer.fast_dev_run and val_trainer.checkpoint_callback:
        checkpoint = torch.load(val_trainer.checkpoint_callback.best_model_path)
        valid_model.load_state_dict(checkpoint["state_dict"])

    # trainer.test() returns List[Dict] of results for each dataloader; we use a single dataloader
    metrics = val_trainer.test(valid_model, datamodule)[0]

    if not val_trainer.fast_dev_run:
        torch.save(
            valid_model.state_dict(), f"saved_models/{encoder_name}_{val_name}.pth"
        )

    return metrics
