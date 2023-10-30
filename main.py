import logging
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import wandb

from src import learn_coles
from src.utils.logging_utils import get_logger
from src.local_validation.local_validation_pipeline import local_target_validation
from src.global_validation.global_validation_pipeline import global_target_validation
from src.generation.learning import train_autoencoder

logging.basicConfig(level=logging.INFO)
logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="config", config_name="config_churn")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    if "model" in cfg:
        model_name: str = hydra_cfg.runtime.choices["model"]
        logger.info(f"Fitting {model_name}...")
        if model_name.startswith("coles"):
            learn_coles(cfg["preprocessing"], cfg["dataset"], cfg["model"])
        elif model_name.startswith("cpc"):
            pass
        elif model_name.startswith("ae"):
            train_autoencoder(
                cfg["preprocessing"],
                cfg["dataset"],
                cfg["model"],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name=}")

    if "validation" in cfg:
        val_name: str = hydra_cfg.runtime.choices["validation"]
        logger.info(f"Validating {val_name}")
        if val_name.startswith("local"):
            res = local_target_validation(cfg["preprocessing"], cfg["validation"])
        elif val_name.startswith("global"):
            res = global_target_validation(cfg["preprocessing"], cfg["validation"])
        else:
            raise ValueError(f"Unsupported validation type: {val_name=}")

        if wandb.run is not None:
            wandb.log({"local_target_table": wandb.Table(dataframe=res)})
            wandb.log({"local_target_stats": wandb.Table(dataframe=res.describe().reset_index())})

        print(res)
        print(res.describe())


if __name__ == "__main__":
    main()
