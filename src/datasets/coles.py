"""
Custom coles datamodule
"""
import torch
import pandas as pd

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices, SampleUniform

class CustomColesDataset(ColesDataset):
    """
    Custom coles dataset inhereted from ptls coles dataset.
    """

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int,
        split_count: int,
        random_min_seq_len: int,
        random_max_seq_len: int,
        deterministic: bool,
        *args,
        col_time: str = "event_time",
        **kwargs
    ):
        """Overrided initialize method, which is suitable for our tasks.

        Args:
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            split_count (int): number of splitting samples
            random_min_seq_len (int): Minimal length of the randomly sampled subsequence
            random_max_seq_len (int): Maximum length of the randomly sampled subsequence
            col_time (str, optional): column name with event time. Defaults to 'event_time'.
        """
        if deterministic:
            splitter = SampleUniform(split_count, (random_min_seq_len + random_max_seq_len) // 2)
        else:
            splitter = SampleSlices(split_count, random_min_seq_len, random_max_seq_len)
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            splitter,
            col_time,
            *args,
            **kwargs
        )
