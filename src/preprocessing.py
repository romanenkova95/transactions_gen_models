from pathlib import Path

from omegaconf import DictConfig
import pickle
from ptls.preprocessing import PandasDataPreprocessor
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


def preprocess(cfg_preprop: DictConfig, local_target_col: str = "") -> pd.DataFrame:
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

    dataset: pd.DataFrame

    if not path_to_preprocessor.exists():
        logger.info(
            "Preprocessor was not saved, so the fitting process will be provided"
        )
        event_time_transformation = (
            "none" if dataset_name == "age" else "dt_to_timestamp"
        )

        cols_numerical = ["amount"]
        if dataset_name != "age" and local_target_col:
            local_target = local_target_col
            cols_numerical += [local_target]

        preprocessor = PandasDataPreprocessor(
            col_id="user_id",
            col_event_time="timestamp",
            event_time_transformation=event_time_transformation,
            cols_category=["mcc_code"],
            cols_numerical=cols_numerical,
            return_records=False,
        )
        dataset = preprocessor.fit_transform(dataframe)  # type: ignore (return_records=False)
        with path_to_preprocessor.open("wb") as file:
            pickle.dump(preprocessor, file)
    else:
        with path_to_preprocessor.open("rb") as file:
            preprocessor: PandasDataPreprocessor = pickle.load(file)
        dataset = preprocessor.transform(dataframe)

    return dataset
