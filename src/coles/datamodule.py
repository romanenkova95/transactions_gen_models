"""
Custom coles datamodule
"""
import numpy as np

import torch

from typing import Optional, List, Dict, Tuple
from functools import reduce
from operator import iadd

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices, AbsSplit
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.padded_batch import PaddedBatch


class SampleAll(AbsSplit):
    """
    Custom sliding window subsequence sampler.
    """

    def __init__(self, seq_len: int, stride: int) -> None:
        """Initialize subsequence sampler. It samples all subsequences from an intial sequence using sliding window.

        Args:
            seq_len (int): desired subsequence length (i.e. sliding window size)
            stride (int): margin between subsequent windows
        """
        self.seq_len = seq_len
        self.stride = stride

    def split(self, dates: np.array) -> List[np.array]:
        """Create list of subsequences indexes.

        Args:
            dates (np.array): array of timestamps with transactions datetimes

        Returns:
            list(np.arrays): list of indexes, corresponding to subsequences;
                             length is num_samples = (num_transactions - seq_len) // stride
        """
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        if date_len <= self.seq_len:
            return [date_range for _ in range(date_len)]

        # crop last 'seq_len' record as we do not have local labels for them
        start_pos = date_range[0 : date_len - self.seq_len : self.stride]
        return [date_range[s : s + self.seq_len] for s in start_pos]


class CustomColesDataset(ColesDataset):
    """
    Custom coles dataset inhereted from ptls coles dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        min_len: int,
        split_count: int,
        random_min_seq_len: int,
        random_max_seq_len: int,
        *args,
        col_time: str = "event_time",
        **kwargs
    ):
        """Overrided initialize method, which is suitable for our tasks

        Args:
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            split_count (int): number of splitting samples
            random_min_seq_len (int): Minimal length of the randomly sampled subsequence
            random_max_seq_len (int): Maximum length of the randomly sampled subsequence
            col_time (str, optional): column name with event time. Defaults to 'event_time'.
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleSlices(split_count, random_min_seq_len, random_max_seq_len),
            col_time,
            *args,
            **kwargs
        )


class CustomColesValidationDataset(ColesDataset):
    """
    Custom coles dataset for local validation pipeline. Items contain all subsequences for each client.
    """

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        min_len: int,
        seq_len: int,
        stride: int,
        *args,
        col_time: str = "event_time",
        local_target_col: Optional[str] = None,
        **kwargs
    ) -> None:
        """Overrided initialize method, which is suitable for local validation pipeline.

        Args:
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            seq_len (int): desired subsequence length (i.e. sliding window size) (parameter of SampleAll sampler)
            stride (int): margin between subsequent windows (parameter of SampleAll sampler)
            col_time (str, optional): column name with event time. Defaults to 'event_time'
            local_target_col (str, optional): if not None, indicates name of the with local targets for each transaction
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleAll(seq_len, stride),
            col_time,
            *args,
            **kwargs
        )
        # keep name of the column with local targets
        self.local_target_col = local_target_col

    def collate_fn(self, batch: List[Dict]) -> Tuple[PaddedBatch, torch.Tensor]:
        """Overwrite collate function to return batch of local targets for a batch of subsequences.
        For each subsequence (i.e. window), which is embedded by CoLES to 1 vector, there is 1 local label.

        Args:
            batch (list[dict]) - batch of ptls format (list with feature dicts)

        Returns:
            if local_target_col is defined:
                a Tuple of:
                    - PaddedBatch object with feature dicts
                    - torch.Tensor with local targets
            else:
                a Tuple of:
                    - PaddedBatch object with feature dicts
                    - torch.Tensor with class labels (client indexes)
        """
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        if self.local_target_col is not None:
            # as CoLES embeds every subsequence (window) to one feature vector,
            # for every window of size 'seq_len', we take local label corresponding for the last time step in this window
            local_targets = padded_batch.payload[self.local_target_col][:, -1]
            return padded_batch, local_targets
        else:
            # repeat standard collate function for ColesDataset
            class_labels = [
                i for i, class_samples in enumerate(batch) for _ in class_samples
            ]
            return padded_batch, torch.LongTensor(class_labels)
