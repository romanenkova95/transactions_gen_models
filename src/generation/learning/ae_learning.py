from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split, DataLoader
from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.augmentations import RandomSlice
from ptls.data_load.utils import collate_feature_dict
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

from src.generation.modules.base import AbsAE
from src.preprocessing import preprocess
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)

def train_autoencoder(
    cfg_preprop: DictConfig, cfg_model: DictConfig
) -> None:
    dataframe: pd.DataFrame
    dataframe = preprocess(cfg_preprop)
    dataframe["mcc_code"] = dataframe["mcc_code"].apply(lambda t: t[t < cfg_model["dataset"]["n_mccs_keep"]])
    dataframe["amount"] = dataframe["amount"].apply(lambda t: t.float())
    dataset = dataframe.to_dict(orient='records')

    dataset = AugmentationDataset(
        MemoryMapDataset(dataset, [SeqLenFilter(cfg_model["dataset"]["min_len"])]),
        [RandomSlice(cfg_model["dataset"]["random_min_seq_len"], cfg_model["dataset"]["random_max_seq_len"])]
    )

    train, val = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        train, collate_fn=collate_feature_dict, **cfg_model.get("train_dl_args", {})
    )

    val_dataloader = DataLoader(
        val, collate_fn=collate_feature_dict, **cfg_model.get("val_dl_args", {})
    )

    module: AbsAE = instantiate(cfg_model["module_ae"], _recursive_=False)(
        encoder_config=cfg_model["encoder"],
        decoder_config=cfg_model["decoder"]
    )

    if "fast_dev_run" not in cfg_model["trainer_args"]:
        lightning_logger = WandbLogger(project="macro_micro_coles")

        cfg = OmegaConf.merge(cfg_model, cfg_preprop)
        lightning_logger.experiment.config.update(OmegaConf.to_container(cfg))
    else:
        lightning_logger = DummyLogger()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=lightning_logger,
        log_every_n_steps=10,
        callbacks=[LearningRateMonitor()],
        **cfg_model["trainer_args"],
    )

    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
