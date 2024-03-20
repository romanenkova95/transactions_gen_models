"""Last token prediction dataset."""

from typing import Any

from .basic_ds import create_basic_dataset
from .target_utils import LastTokenTarget, collate_fn_with_targets


def create_last_token_dataset(
    data: Any,
    deterministic: bool,
    min_len: int,
    random_min_seq_len: int,
    random_max_seq_len: int,
    window_size: int,
    window_step: int,
    target_seq_col: str,
    drop_last: bool,
):
    """Initialize dataset, which returns tuple of batch & last element of target_seq_col.

    Optionally, drop last elements of sequence columns to prevent data leak.

    Args:
    ----
        data (Any):
            Data, compatible with ptls.datasets
        deterministic (bool):
            Whether to sample randomly from the dataset (train behaviour),
            or to use a deterministic sliding-window strategy (evaluation behaviour).
        min_len (int):
            anything longer than this is filtered out
        random_min_seq_len (int):
            minimum len of sampled subsequence
        random_max_seq_len (int):
            maximum len of sampled subsequence
        window_size (int):
            size of eval window
        window_step (int):
            step of eval window
        target_seq_col (str):
            Name of target sequence column.
        drop_last (bool):
            Whether to drop last elements of sequence column
    """
    augmentations = [LastTokenTarget(target_seq_col, drop_last)]
    return create_basic_dataset(
        data=data,
        deterministic=deterministic,
        min_len=min_len,
        random_min_seq_len=random_min_seq_len,
        random_max_seq_len=random_max_seq_len,
        window_size=window_size,
        window_step=window_step,
        f_augmentations=augmentations,
        collate_fn=collate_fn_with_targets,
    )
