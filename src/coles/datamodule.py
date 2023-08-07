from omegaconf import DictConfig

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleUniform


class MyColesDataset(ColesDataset):

    def __init__(
        self,
        data: list[dict],
        coles_conf: DictConfig,
        col_time: str = 'event_time',
        *args, **kwargs
    ):

        min_len: int = coles_conf['data']['min_len']
        split_count: int = coles_conf['data']['split_count']
        seq_len: int = coles_conf['data']['seq_len']

        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleUniform(split_count, seq_len),
            col_time,
            *args, **kwargs
        )
