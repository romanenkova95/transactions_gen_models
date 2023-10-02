from hydra.utils import instantiate

from joblib import Memory
import pandas as pd

from ptls.preprocessing.base import ColTransformer

@Memory("cache", verbose=5).cache
def preprocess(cfg: dict) -> list[dict]:
    """Preprocess data according to given config. Caches function result using joblib to cache directory

    Args:
        cfg (dict): loaded OmegaConf config, converted to dict for compatibility with joblib. Needs fields:
            - source: .parquet file to read the dataframe from
            - transforms: sequence of ColTransformer/scikit-learn transforms of the pandas dataframe
        Other fields are allowed and ignored.

    Returns:
        list[dict]: FeatureDict, compatible with ptls
    """
    dataframe: pd.DataFrame = pd.read_parquet(cfg["source"])

    transform: ColTransformer
    for transform in instantiate(cfg["transforms"]):
        dataframe = transform.fit_transform(dataframe) # type: ignore

    return dataframe.to_dict(orient='records')
