"""Module with the coles dataset, which includes a splitter."""
import torch
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import AbsSplit, SampleUniform


class CustomColesDatasetWithSplitter(ColesDataset):
    """CoLES dataset with splitter as an argument."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        split_count: int,
        min_len: int,
        splitter: AbsSplit,
        deterministic: bool,
        *args,
        col_time: str = "event_time",
        **kwargs,
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict[str, torch.Tensor]]): the raw datset.
            split_count (int): number of splits per client.
            min_len (int): minimum length of transaction sequence (smaller sequences discarded).
            splitter (AbsSplit): splitter to use when splitting the user sequences.
            deterministic (bool): whether to use deterministic sampling, for evaluation.
            *args: additional positional arguments, passed to the ColesDataset class.
            col_time (str, optional): name of the time column. Defaults to "event_time".
            **kwargs: additional keyword arguments, passed to the ColesDataset class.
        """
        if deterministic:
            splitter = SampleUniform(split_count, 2 * min_len)
        else:
            splitter = splitter

        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            splitter,
            col_time,
            *args,
            **kwargs,
        )
