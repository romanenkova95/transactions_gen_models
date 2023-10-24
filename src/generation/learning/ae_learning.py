"""Autoencoder training script"""

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ptls.frames import PtlsDataModule
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger

from src.generation.modules.base import AbsAE
from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess

logger = get_logger(name=__name__)


def train_autoencoder(
    cfg_preprop: DictConfig, cfg_dataset: DictConfig, cfg_model: DictConfig
) -> None:
    """Train autoencoder, specified in the model config, on the data, specified by the preprocessing config.

    Args:
        cfg_preprop (DictConfig):
            the preprocessing config, for specifications see preprocess.
        cfg_dataset (DictConfig):
            the dataset config, to be partially instantiated, and later supplied the data argument,
            a list of dicts of features, outputted by preprocess.
        cfg_model (DictConfig):
            the model config, which should contain:
                - encoder: nn.Module of encoder
                - decoder: nn.Module of decoder
                - module_ae: pl.Module, used to train & validate the model, subclass of AbsAE
                - trainer_args: arguments to pass to pl.Trainer
                - datamodule_args: arguments to pass when constructing the ptls datamodule. Optional, defaults to {}
                - split_seed: randomness seed to use when splitting records. Optional, defaults to 42
    """
    train, val, test = preprocess(cfg_preprop)

    ds_factory = instantiate(cfg_dataset, _partial_=True)
    datamodule = PtlsDataModule(
        train_data=ds_factory(train),
        valid_data=ds_factory(val),
        test_data=ds_factory(test),
        **cfg_model.get("datamodule_args", {}),
    )

    module: AbsAE = instantiate(cfg_model["module_ae"], _recursive_=False)(
        encoder_config=cfg_model["encoder"], decoder_config=cfg_model["decoder"]
    )

    callbacks = []

    if "fast_dev_run" not in cfg_model["trainer_args"]:
        lightning_logger = WandbLogger(project="macro_micro_coles")

        cfg_dict = {
            "preprocessing": OmegaConf.to_container(cfg_preprop),
            "dataset": OmegaConf.to_container(cfg_dataset),
            "model": OmegaConf.to_container(cfg_model)
        }

        lightning_logger.experiment.config.update(cfg_dict)
        callbacks.append(LearningRateMonitor())
    else:
        lightning_logger = DummyLogger()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=lightning_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        **cfg_model["trainer_args"],
    )

    trainer.fit(module, datamodule=datamodule)
