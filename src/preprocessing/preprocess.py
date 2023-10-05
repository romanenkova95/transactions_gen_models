from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from joblib import Memory
import pandas as pd
from sklearn.model_selection import train_test_split

from ptls.preprocessing.base import ColTransformer


def preprocess(cfg: DictConfig) -> tuple[list[dict], list[dict], list[dict]]:
    """Preprocess data according to given config. Caches function result using joblib to cache directory

    Args:
        cfg (dict): loaded OmegaConf config, converted to dict for compatibility with joblib. Needs fields:
            - source: .parquet file to read the dataframe from
            - transforms: sequence of ColTransformer/scikit-learn transforms of the pandas dataframe
        Other fields are allowed and ignored.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: 
            train/val/test FeatureDicts, compatible with ptls
    """
    def _preprocess(cfg: dict):
        dataframe: pd.DataFrame = pd.read_parquet(cfg["source"])

        transform: ColTransformer
        for transform in instantiate(cfg["transforms"]):
            dataframe = transform.fit_transform(dataframe) # type: ignore

        return dataframe.to_dict(orient='records')
    
    if "cache_dir" in cfg:
        memory = Memory("cache", verbose=5)
        _preprocess = memory.cache(_preprocess) # type: ignore
    
    data = _preprocess(OmegaConf.to_container(cfg)) # type: ignore
    val_test_size = cfg["val_size"] + cfg["test_size"]
    train, val_test = train_test_split(data, test_size=val_test_size, random_state=cfg["random_state"])
    val, test = train_test_split(val_test, test_size=cfg["test_size"] / (val_test_size), random_state=cfg["random_state"])
    
    return train, val, test
