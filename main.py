import logging
from typing import Optional

import hydra
from omegaconf import DictConfig

from src import learn_coles
from src.local_validation.local_validation_pipeline import local_target_validation
from src.global_validation.global_validation_pipeline import global_target_validation

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_churn")
def main(cfg: Optional[DictConfig] = None) -> None:
    if cfg["model"]["name"].startswith("coles"):
        learn_coles(cfg["dataset"], cfg["model"])
        #res = global_target_validation(cfg["dataset"], cfg["validation"])
        res = local_target_validation(cfg["dataset"], cfg["validation"])
    elif cfg["model"]["name"].startswith("cpc"):
        pass

    print(res)

if __name__ == "__main__":
    main()
