import logging
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src.global_validation import global_target_validation

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_validation")
def main(cfg: Optional[DictConfig] = None) -> None:
    cfg_preprop = cfg["dataset"]
    cfg_validation = cfg["validation"]

    logger: WandbLogger = instantiate(cfg_validation["logger"])

    res = global_target_validation(cfg_preprop, cfg_validation)
    aggregated_res = res.agg(["mean", "std"])
    print(aggregated_res)

    logger.experiment.log({"AUROC": aggregated_res["AUROC"]["mean"]})
    logger.experiment.log({"PR-AUC": aggregated_res["PR-AUC"]["mean"]})
    logger.experiment.log({"Accuracy": aggregated_res["Accuracy"]["mean"]})
    logger.experiment.log({"F1Score": aggregated_res["F1Score"]["mean"]})


if __name__ == "__main__":
    main()
