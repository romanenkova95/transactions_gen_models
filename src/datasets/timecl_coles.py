import numpy as np
import torch

from typing import List
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import AbsSplit, SampleUniform

from .splitters import TimeCLSampler


class TimeCLColesDataset(ColesDataset):
    """
    CoLES dataset with TimeCL sampler.
    """

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int,
        max_len: int,
        split_count: int,
        llambda: float,
        rlambda: float,
        deterministic: bool,
        *args,
        col_time: str = "event_time",
        **kwargs
    ):
        if deterministic:
            splitter = SampleUniform(split_count, (min_len + max_len) // 2)
        else:
            splitter = TimeCLSampler(min_len, max_len, llambda, rlambda, split_count)

        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            splitter,
            col_time,
            *args,
            **kwargs
        )
