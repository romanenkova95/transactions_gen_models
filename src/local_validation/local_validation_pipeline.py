"""Local targets validation script. """

from pathlib import Path
import warnings
from hydra.utils import instantiate, call
from omegaconf import DictConfig

import pandas as pd
import pytorch_lightning

import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

from ptls.frames import PtlsDataModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess
from .local_validation_model import LocalValidationModelBase

def local_target_validation(cfg: DictConfig, encoder_name: str, val_name) -> pd.DataFrame:
    """Full pipeline for the sequence encoder local validation. 

    Args:
        cfg_preprop (DictConfig):    Dataset config (specified in the 'config/dataset')
        cfg_validation (DictConfig): Validation config (specified in the 'config/validation')
    
    Returns:
        results (pd.DataFrame):      Dataframe with test metrics for each run
    """
    logger = get_logger(name=__name__)
    train, val, test = preprocess(cfg["preprocessing"])

    logger.info("Instantiating the sequence encoder")
    # load pretrained sequence encoder
    sequence_encoder: SeqEncoderContainer = instantiate(
        cfg["encoder"], 
        is_reduce_sequence=cfg["validation"]["is_reduce_sequence"]
    )
    encoder_state_dict_path = Path(f"saved_models/{encoder_name}.pth")
    if encoder_state_dict_path.exists():
        sequence_encoder.load_state_dict(torch.load(encoder_state_dict_path))
    else:
        warnings.warn("No encoder state dict found! Validating without pre-loading state-dict...")

    train_dataset = call(cfg["validation"]["dataset"], data=train, deterministic=False)
    val_dataset = call(cfg["validation"]["dataset"], data=val, deterministic=True)
    test_dataset = call(cfg["validation"]["dataset"], data=test, deterministic=True)

    datamodule: PtlsDataModule = instantiate(
        cfg["validation"]["datamodule"],
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
    )

    results = []
    for i in range(cfg["validation"]["n_runs"]):
        logger.info(f'Training LocalValidationModel. Run {i+1}/{cfg["validation"]["n_runs"]}')

        seed_everything(i)

        valid_model: LocalValidationModelBase = instantiate(
            cfg["validation"]["module"],
            backbone=sequence_encoder 
        )

        val_trainer: Trainer = instantiate(cfg["validation"]["trainer"])
        
        # Make wandb log runs to different metric graphs
        if isinstance(val_trainer.logger, WandbLogger):
            val_trainer.logger._prefix = f"{val_name}{i}"

        val_trainer.fit(valid_model, datamodule)
        torch.save(valid_model.state_dict(), f'saved_models/{val_name}_validation_head_{i}.pth')

        # trainer.test() returns List[Dict] of results for each dataloader; we use a single dataloader
        metrics = val_trainer.test(valid_model, datamodule)[0]
        results.append(metrics)

    return pd.DataFrame(results)
