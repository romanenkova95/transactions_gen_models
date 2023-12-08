"""File with tools for creation of the time diff dataset."""
from typing import Any

from .basic_ds import create_basic_dataset
from .target_utils import TimeDiffTarget, collate_fn_with_targets


def create_time_diff_dataset(
    data: Any,
    deterministic: bool,
    min_len: int,
    random_min_seq_len: int,
    random_max_seq_len: int,
    window_size: int,
    window_step: int,
    time_col: str = "event_time",
):
    """Initialize dataset, which returns tuple of batch & time until next transaction.

    Args:
    ----
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
        window_size (int):
            size of eval window
        window_step (int):
            step of eval window
        time_col (str):
            Name of column with timestamps
    """
    f_augmentations = [TimeDiffTarget(time_col)]
    return create_basic_dataset(
        data=data,
        deterministic=deterministic,
        min_len=min_len,
        random_min_seq_len=random_min_seq_len,
        random_max_seq_len=random_max_seq_len,
        window_size=window_size,
        window_step=window_step,
        f_augmentations=f_augmentations,
        collate_fn=collate_fn_with_targets,
    )
