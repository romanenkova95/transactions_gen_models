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


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig) -> None:
    run(cfg)

def run(cfg: DictConfig):
    if not cfg:
        raise ValueError("Empty or no config! Please run with --config-name argument with valid config name")
    
    data = preprocess(cfg["preprocessing"])
    
    # Disable WandB logging in fast_dev_run mode
    if os.environ["FAST_DEV_RUN"]:
        os.environ["WANDB_MODE"] = "disabled"
    
    hydra_cfg = HydraConfig.get()
    encoder_name: str = hydra_cfg.runtime.choices["encoder"]
    if "module" in cfg:
        module_name: str = hydra_cfg.runtime.choices["module"]
        logger.info(f"Fitting {module_name}...")
        learn(data, cfg, encoder_name)
    
    if "validation" in cfg:
        for val_name, cfg_validation in cfg["validation"].items():
            logger.info(f"{val_name} validation for {encoder_name}")
            if val_name.startswith("global_target"):
                res = global_target_validation(
                    data,
                    cfg["encoder"], 
                    cfg_validation,
                    encoder_name
                )
            else:
                res = local_target_validation(
                    data,
                    cfg["encoder"], 
                    cfg_validation,
                    encoder_name,
                    val_name
                )

            if wandb.run is not None:
                wandb.log({f"{val_name}_local_target_table": wandb.Table(dataframe=res)})
                wandb.log({f"{val_name}_local_target_stats": wandb.Table(dataframe=res.describe().reset_index())})
                

            print(res)
            print(res.describe())
    
    wandb.finish()
        

if __name__ == "__main__":
    main()
