"""Module with function for creating the basic dataset."""
from typing import Any, Callable, Optional

from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter

from .random_crop import RandomCropDataset
from .sliding_window import SlidingWindowDataset


def create_basic_dataset(
    data: Any,
    deterministic: bool,
    min_len: int,
    random_min_seq_len: int,
    random_max_seq_len: int,
    window_size: int,
    window_step: int,
    f_augmentations: Optional[list[Callable]] = None,
    collate_fn: Optional[Callable] = None,
):
    """Initialize dataset.

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
        f_augmentations (list[callable]):
            Optional post-slice augmentations of resulting feature dict.
        collate_fn (callable):
            collate_fn arg to pass to dataloader, defaults to None for collate_feature_dict.
    """
    dataset = MemoryMapDataset(data, [SeqLenFilter(min_len)])
    if deterministic:
        return SlidingWindowDataset(
            data=dataset,
            window_size=window_size,
            window_step=window_step,
            f_augmentations=f_augmentations,
            collate_fn=collate_fn,
        )
    else:
        return RandomCropDataset(
            data=dataset,
            random_min_seq_len=random_min_seq_len,
            random_max_seq_len=random_max_seq_len,
            f_augmentations=f_augmentations,
            collate_fn=collate_fn,
        )
