"""Next token prediction dataset"""
from typing import Any

import torch

from ptls.data_load.utils import collate_feature_dict

from .basic_ds import create_basic_dataset
from .transforms import LastTokenTarget


def collate_fn_with_targets(batch):
    x, y = zip(*batch)
    return collate_feature_dict(x), torch.stack(y)


def create_next_token_dataset(
    data: Any,
    deterministic: bool,
    min_len: int, 
    random_min_seq_len: int, 
    random_max_seq_len: int, 
    target_seq_col: str,
    window_size: int,
    window_step: int,
    ):
    """Initialize dataset

    Args:
        data (Any):
            Data, compatible with ptls.datasets
        deterministic (bool):
            Whether to sample randomly from the dataset (train behaviour),
            or to use a deterministic sliding-window strategy (evaluation behaviour).
        min_len (int):
            minimum sequence length (anything longer than this is filtered out)
        random_min_seq_len (int):
            minimum len of sampled subsequence
        random_max_seq_len (int):
            maximum len of sampled subsequence
        target_seq_col (str):
            Name of target sequence column.
    """
    augmentations = [LastTokenTarget(target_seq_col)]
    return create_basic_dataset(
        data=data, 
        deterministic=deterministic,
        min_len=min_len,
        random_min_seq_len=random_min_seq_len,
        random_max_seq_len=random_max_seq_len,
        window_size=window_size,
        window_step=window_step,
        f_augmentations=augmentations,
        collate_fn=collate_fn_with_targets
    )
