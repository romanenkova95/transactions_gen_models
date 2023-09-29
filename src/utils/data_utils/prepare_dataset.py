"""Dataset preparation script"""
from pathlib import Path
import pickle

import logging
from omegaconf import DictConfig

import pandas as pd

from ptls.preprocessing import PandasDataPreprocessor

from src.custom_preprocessing import CustomDatetimeNormalization


def prepare_dataset(cfg_preprop: DictConfig, logger: logging.Logger) -> list[dict]:
    """Prepares dataset.
    
    Args:
        cfg_preprop (DictConfig): Dataset config (specified in the 'config/dataset')
        logger (logging.Logger):  Logger

    Returns:
        dataset (list): Dataset int ptls format (list of dicts)
    """
    dataframe = pd.read_parquet(
        Path(cfg_preprop["dir_path"]).joinpath(cfg_preprop["train_file_name"])
    )
    logger.info("dataframe initialized")

    dataset_name = cfg_preprop["name"]

    path_to_preprocessor = Path(cfg_preprop["coles"]["pandas_preprocessor"]["dir_path"])
    if not path_to_preprocessor.exists():
        logger.warning("Preprocessor directory does not exist. Creating")
        path_to_preprocessor.mkdir(parents=True)
    path_to_preprocessor = path_to_preprocessor.joinpath(
        cfg_preprop["coles"]["pandas_preprocessor"]["name"]
    )

    if not path_to_preprocessor.exists():
        logger.info(
            "Preprocessor was not saved, so the fitting process will be provided"
        )

        if cfg_preprop["datetime_transformation"] is None:
        # default datetime stransformation
            event_time_transformation = (
                "none" if dataset_name == "age" else "dt_to_timestamp"
            )

            preprocessor = PandasDataPreprocessor(
                col_id="user_id",
                col_event_time="timestamp",
                event_time_transformation=event_time_transformation,
                cols_category=["mcc_code"],
                cols_numerical=["amount"],
                cols_first_item=[
                    "global_target"
                ],  # global target is duplicated, use 1st value
                return_records=True,
            )
        
        elif cfg_preprop["datetime_transformation"] == "normalize":
        # normalize times
            min_timestamp = int(dataframe["timestamp"].min().timestamp())

            datetime_transformer = CustomDatetimeNormalization(
                min_timestamp=min_timestamp, col_name_original="timestamp"
            )

            preprocessor = PandasDataPreprocessor(
                col_id="user_id",
                col_event_time=datetime_transformer,
                cols_category=["mcc_code"],
                category_transformation="frequency",
                cols_numerical=["amount"],
                cols_first_item=["global_target"],
                return_records=True,
            )
        else:
            raise ValueError(f"Unknown datetime transformation {cfg_preprop['datetime_transformation']}.")

        dataset = preprocessor.fit_transform(dataframe)

        with path_to_preprocessor.open("wb") as file:
            pickle.dump(preprocessor, file)
    else:
        with path_to_preprocessor.open("rb") as file:
            preprocessor: PandasDataPreprocessor = pickle.load(file)
        dataset = preprocessor.transform(dataframe)

    return dataset
