import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
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

    # Setup names & log config
    hydra_cfg = HydraConfig.get()
    preproc_name: str = hydra_cfg.runtime.choices["preprocessing"]
    backbone_name: str = hydra_cfg.runtime.choices["backbone"]
    val_names: list[str] = cfg.get("validation", {}).keys()
    lightning_logger_name = hydra_cfg.runtime.choices.get("logger", "tensorboard")
    if lightning_logger_name == "wandb":
        wandb.init(
            project="macro_micro_coles", 
            config=OmegaConf.to_container(cfg), # type: ignore
            tags=[preproc_name, backbone_name, *val_names]
        )
    # CometML TODO (optionally): add experiment tagging

    data = preprocess(cfg["preprocessing"])
    seed = seed_everything(cfg.get("seed"))
    experiment_name = f"{backbone_name}_{preproc_name}_{seed}"
    
    if cfg["pretrain"]:
        logger.info(f"Experiment {experiment_name}...")
        learn(
            data=data,
            backbone_cfg=cfg["backbone"],
            logger_cfg=cfg["logger"],
            encoder_save_name=experiment_name,
        )

    for val_name, cfg_validation in cfg.get("validation", {}).items():
        reset_seed() # get seed from os.environ["PL_GLOBAL_SEED"]
        logger.info(f"{val_name} validation for {experiment_name}")
        if val_name.startswith("global_target"):
            res = global_target_validation(
                data=data, 
                cfg_encoder=cfg["backbone"]["encoder"], 
                cfg_validation=cfg_validation, 
                encoder_name=experiment_name
            )
            
            res = {"global_target" + k: v for k, v in res.items()}
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
        if lightning_logger_name == "wandb":
            wandb.log(res)
        
        # CometML TODO (optionally): add global validation logging
            
    if lightning_logger_name == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
