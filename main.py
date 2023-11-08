import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import wandb

from src import learn
from src.preprocessing import preprocess
from src.utils.logging_utils import get_logger
from src.local_validation.local_validation_pipeline import local_target_validation
from src.global_validation.global_validation_pipeline import global_target_validation

logging.basicConfig(level=logging.INFO)
logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="config", config_name="master.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


def run(cfg: DictConfig):
    if not cfg:
        raise ValueError(
            "Empty or no config! Please run with --config-name argument with valid config name"
        )

    data = preprocess(cfg["preprocessing"])

    hydra_cfg = HydraConfig.get()
    experiment_name: str = hydra_cfg.runtime.choices["backbone"]
    module_name: str = hydra_cfg.runtime.choices["module"]
    if cfg["pretrain"]:
        logger.info(f"Fitting {module_name}...")
        learn(
            data=data,
            backbone_cfg=cfg["backbone"],
            logger_cfg=cfg["logger"],
            encoder_save_name=experiment_name,
        )

    for val_name, cfg_validation in cfg.get("validation", {}).items():
        logger.info(f"{val_name} validation for {experiment_name}")
        if val_name.startswith("global_target"):
            res = global_target_validation(
                data, cfg["backbone"]["encoder"], cfg_validation, experiment_name
            )
        else:
            res = local_target_validation(
                data=data,
                cfg_encoder=cfg["backbone"]["encoder"],
                cfg_validation=cfg_validation,
                cfg_logger=cfg["logger"],
                encoder_name=experiment_name,
                val_name=val_name,
            )

        print(res)

    wandb.finish()


if __name__ == "__main__":
    main()
