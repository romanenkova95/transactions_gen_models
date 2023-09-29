import logging
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import learn_coles
from src.preprocessing import preprocess
from src.generation.learning.ae_learning import train_autoencoder

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_churn")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    model_name: str = hydra_cfg.runtime.choices["model"]

    if model_name.startswith("coles"):
        learn_coles(cfg["preprocessing"], cfg["dataset"], cfg["model"])
    elif model_name.startswith("cpc"):
        pass
    elif model_name.startswith("ae"):
        train_autoencoder(cfg["preprocessing"], cfg["dataset"], cfg["model"])
    else:
        raise ValueError(f"Unsupported model type: {model_name=}")


if __name__ == "__main__":
    main()
