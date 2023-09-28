from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

from joblib import Memory
import pandas as pd

from ptls.preprocessing.base import ColTransformer


@Memory("cache").cache()
def preprocess(cfg: DictConfig) -> List[Dict]:
    """Preprocess data according to given config

    Args:
        cfg (DictConfig): loaded OmegaConf config. Needs fields:
            - source: .parquet file to read the dataframe from
            - transforms: sequence of ColTransformer/scikit-learn transforms of the pandas dataframe

    Returns:
        List[Dict]: FeatureDict, compatible with ptls
    """
    dataframe: pd.DataFrame = pd.read_parquet(cfg["source"])

    transform: ColTransformer
    for transform in instantiate(cfg["transforms"]):
        dataframe = transform.fit_transform(dataframe) # type: ignore

    return dataframe.to_dict(orient='records')
