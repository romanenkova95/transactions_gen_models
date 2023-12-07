import torch

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleUniform

from ptls.frames.coles.split_strategy import AbsSplit


class CustomColesDatasetWithSplitter(ColesDataset):
    """
    CoLES dataset with splitter as an argument.
    """

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        split_count: int,
        min_len: int,
        splitter: AbsSplit,
        deterministic: bool,
        *args,
        col_time: str = "event_time",
        **kwargs
    ):
        if deterministic:
            splitter = SampleUniform(split_count, 2 * min_len)
        else:
            splitter = splitter

        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            splitter,
            col_time,
            *args,
            **kwargs
        )
