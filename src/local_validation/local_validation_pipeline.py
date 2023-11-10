"""Local targets validation script. """

from pathlib import Path
import warnings
from hydra.utils import instantiate, call
from omegaconf import DictConfig

import pandas as pd

import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

from ptls.frames import PtlsDataModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from src.utils.create_trainer import create_trainer

from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess
from .local_validation_model import LocalValidationModelBase


def local_target_validation(
    data: tuple[list[dict], list[dict], list[dict]],
    cfg_encoder: DictConfig,
    cfg_validation: DictConfig,
    cfg_logger: DictConfig,
    encoder_name: str,
    val_name: str,
) -> dict[str, float]:
    """Full pipeline for the sequence encoder local validation.

    Args:
        data (tuple[list[dict], list[dict], list[dict]]):
            train, val & test sets
        cfg_encoder (DictConfig):
            Encoder config (specified in 'config/encoder')
        cfg_validation (DictConfig):
            Validation config (specified in 'config/validation')
        encoder_name (str):
            Name of used encoder (for logging & saving)
        val_name (str):
            Name of validation (for logging & saving)

    Returns:
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

    train_dataset = call(cfg_validation["dataset"], data=train, deterministic=False)
    val_dataset = call(cfg_validation["dataset"], data=val, deterministic=True)
    test_dataset = call(cfg_validation["dataset"], data=test, deterministic=True)

    datamodule: PtlsDataModule = instantiate(
        cfg_validation["datamodule"],
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
    )

    logger.info(f"Training LocalValidationModel")

    seed_everything()
    valid_model: LocalValidationModelBase = instantiate(
        cfg_validation["module"], backbone=sequence_encoder
    )

    val_trainer: Trainer = create_trainer(
        logger=cfg_logger,
        metric_name=valid_model.metric_name,
        **cfg_validation["trainer"],
    )

    # Make wandb log runs to different metric graphs
    if isinstance(val_trainer.logger, WandbLogger):
        val_trainer.logger._prefix = f"{val_name}"

    val_trainer.fit(valid_model, datamodule)
    if not val_trainer.fast_dev_run and val_trainer.checkpoint_callback:
        checkpoint = torch.load(val_trainer.checkpoint_callback.best_model_path)
        valid_model.load_state_dict(checkpoint["state_dict"])

    torch.save(valid_model.state_dict(), f"saved_models/{val_name}_validation_head.pth")
    # trainer.test() returns List[Dict] of results for each dataloader; we use a single dataloader
    metrics = val_trainer.test(valid_model, datamodule)[0]

    return metrics
