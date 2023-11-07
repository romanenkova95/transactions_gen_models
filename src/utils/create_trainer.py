from ast import literal_eval
import os
from typing import Optional
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def create_trainer(
    logger: Optional[DictConfig] = None,
    metric_name: Optional[str] = None,
    checkpointing: bool = True,
    **kwargs,
):
    # Instantiate callbacks
    instantiated_callbacks = []
    for callback in kwargs.pop("callbacks", []):
        cls: str = callback["_target_"].split(".")[-1]
        if cls == "EarlyStopping":
            if not metric_name:
                raise ValueError(f"Metric name required for {callback}")

            callback_instance = instantiate(
                callback, monitor=metric_name, mode="min" if "loss" in metric_name else "max"
            )
        else:
            callback_instance = instantiate(callback)

        instantiated_callbacks.append(callback_instance)

    if checkpointing and metric_name:
        instantiated_callbacks.append(
            ModelCheckpoint(monitor=metric_name, mode="min" if "loss" in metric_name else "max")
        )

    # Instantiate logger
    logger_instance = instantiate(logger) if logger else True

    # Set some reasonable defaults
    kwargs["devices"] = kwargs.get("devices", 1)
    kwargs["accelerator"] = kwargs.get("accelerator", "gpu")
    kwargs["precision"] = kwargs.get("precision", 16)
    fast_dev_run = literal_eval(str(os.getenv("FAST_DEV_RUN")))

    # Disable WandB logging in fast_dev_run mode
    if fast_dev_run:
        os.environ["WANDB_MODE"] = "disabled"

    return Trainer(
        logger=logger_instance,
        callbacks=instantiated_callbacks,
        fast_dev_run=fast_dev_run,
        **kwargs,
    )
