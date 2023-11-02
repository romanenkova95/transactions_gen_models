import logging
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import wandb

from src import learn
from src.utils.logging_utils import get_logger
from src.local_validation.local_validation_pipeline import local_target_validation
from src.global_validation.global_validation_pipeline import global_target_validation

logging.basicConfig(level=logging.INFO)
logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig) -> None:
    if not cfg:
        raise ValueError("Empty or no config! Please run with --config-name argument with valid config name")
    
    hydra_cfg = HydraConfig.get()
    encoder_name: str = hydra_cfg.runtime.choices["encoder"]
    if "module" in cfg:
        module_name: str = hydra_cfg.runtime.choices["module"]
        logger.info(f"Fitting {module_name}...")
        learn(cfg, encoder_name)
    
    if "validation" in cfg:
        val_name: str = hydra_cfg.runtime.choices["validation"]
        logger.info(f"{val_name} validation for {encoder_name}")
        if val_name in {"event_type", "event_time", "local_target"}:
            res = local_target_validation(cfg, encoder_name, val_name)
        elif val_name == "global_target":
            res = global_target_validation(cfg, encoder_name)
        else:
            raise ValueError(f"Unsupported validation type: {val_name=}")

        if wandb.run is not None:
            wandb.log({"local_target_table": wandb.Table(dataframe=res)})
            wandb.log({"local_target_stats": wandb.Table(dataframe=res.describe().reset_index())})

        print(res)
        print(res.describe())


if __name__ == "__main__":
    main()
