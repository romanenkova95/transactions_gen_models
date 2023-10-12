import logging
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import learn_coles
from src.local_validation.local_validation_pipeline import local_target_validation
from src.global_validation.global_validation_pipeline import global_target_validation
from src.generation.learning import train_autoencoder

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_churn")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    model_name: str = hydra_cfg.runtime.choices["model"]
    if model_name.startswith("coles"):
        learn_coles(cfg["preprocessing"], cfg["dataset"], cfg["model"])
        # res = global_target_validation(cfg["dataset"], cfg["validation"])
        res = local_target_validation(cfg["preprocessing"], cfg["validation"])
    elif model_name.startswith("cpc"):
        pass
    elif model_name.startswith("ae"):
        train_autoencoder(cfg["preprocessing"], cfg["dataset"], cfg["model"])
    else:
        raise ValueError(f"Unsupported model type: {model_name=}")


if __name__ == "__main__":
    main()
