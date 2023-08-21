"""
Custom coles datamodule
"""
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleUniform


class CustomColesDataset(ColesDataset):
    """
    Custom coles dataset inhereted from ptls coles dataset.
    """

    def __init__(
        self,
        data: list[dict],
        min_len: int,
        split_count: int,
        seq_len: int,
        *args,
        col_time: str = 'event_time',
        **kwargs
    ):
        """Overrided initialize method, which is suitable for our tasks

        Args:
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            split_count (int): number of splitting samples
            seq_len (int): length of the uniform sampled subsequences
            col_time (str, optional): column name with event time. Defaults to 'event_time'.
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleUniform(split_count, seq_len),
            col_time,
            *args, **kwargs
        )
