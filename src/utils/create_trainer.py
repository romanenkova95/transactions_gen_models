"""File with the logic of trainer creation."""
import os
from ast import literal_eval
from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def create_trainer(
    logger: Optional[DictConfig] = None,
    metric_name: Optional[str] = None,
    checkpointing: bool = True,
    **kwargs,
):
    """Create trainer, handling early stopping, some nice defaults, etc.

    Args:
    ----
        logger (Optional[DictConfig], optional):
            the loggger to use. Defaults to None, to use TB.
        metric_name (Optional[str], optional):
            the metric name to monitor with EarlyStopping. Defaults to None.
        checkpointing (bool, optional):
            whether to use checkpointing. Defaults to True.
        **kwargs:
            additional kwargs, passed when initializing the pl Trainer.

    Raises:
    ------
        ValueError: no metric name provided, but found early stopping.

    Returns:
    -------
        Trainer: resulting lightning trainer.
    """
    # Instantiate callbacks
    instantiated_callbacks = []
    for callback in kwargs.pop("callbacks", []):
        cls: str = callback["_target_"].split(".")[-1]
        if cls == "EarlyStopping":
            if not metric_name:
                raise ValueError(f"Metric name required for {callback}")

            callback_instance = instantiate(
                callback,
                monitor=metric_name,
                mode="min" if "loss" in metric_name else "max",
            )
        else:
            callback_instance = instantiate(callback)

        instantiated_callbacks.append(callback_instance)

    if checkpointing and metric_name:
        instantiated_callbacks.append(
            ModelCheckpoint(
                monitor=metric_name, mode="min" if "loss" in metric_name else "max"
            )
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
