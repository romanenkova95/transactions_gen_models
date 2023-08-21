from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleUniform


class CustomColesDataset(ColesDataset):

    def __init__(
        self,
        data: list[dict],
        min_len: int,
        split_count: int,
        seq_len: int,
        col_time: str = 'event_time',
        *args, **kwargs
    ):
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleUniform(split_count, seq_len),
            col_time,
            *args, **kwargs
        )
