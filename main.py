import logging
from typing import Optional

import hydra
from omegaconf import DictConfig

from src import learn_coles

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Optional[DictConfig] = None) -> None:
    if cfg["model"]["name"] == "coles":
        learn_coles(cfg["dataset"], cfg["model"])


if __name__ == "__main__":
    main()
