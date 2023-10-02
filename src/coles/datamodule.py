"""
Custom coles datamodule
"""
import random
from functools import reduce
from operator import iadd
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.padded_batch import PaddedBatch
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import AbsSplit, SampleSlices


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

    def split(self, dates: np.ndarray) -> List[np.ndarray]:
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


class TimeCLSampler(AbsSplit):
    """
    TimeCL sampler implementation, ptls-style.
    For details, see Algorithm 1 from the paper:
        http://mesl.ucsd.edu/pubs/Ranak_AAAI2023_PrimeNet.pdf
    Args:
        min_len (int): minimum subswequence length
        max_len (int): maximum subsequence lentgh
        llambda (float): lower bound for lambda valu
        rlambda (float): upper bound for lambda value
        split_count (int): number of generated subsequences
    """

    def __init__(
        self,
        min_len: int,
        max_len: int,
        llambda: float,
        rlambda: float,
        split_count: int,
    ) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.llambda = llambda
        self.rlambda = rlambda
        self.split_count = split_count

    def split(self, dates: np.ndarray) -> List[list]:
        """Create list of subsequences indexes.

        Args:
            dates (np.array): array of timestamps with transactions datetimes

        Returns:
            list(np.arrays): list of indexes, corresponding to subsequences
        """
        date_len = dates.shape[0]
        idxs = np.arange(date_len)
        if date_len <= self.min_len:
            return [idxs for _ in range(self.split_count)]

        time_deltas = np.concatenate(
            (
                [dates[1] - dates[0]],
                0.5 * (dates[2:] - dates[:-2]),
                [dates[-1] - dates[-2]],
            )
        )

        idxs = sorted(idxs, key=lambda idx: time_deltas[idx])

        dense_timestamps, sparse_timestamps = (
            idxs[: date_len // 2],
            idxs[date_len // 2 :],
        )
        l_dense, l_sparse = len(dense_timestamps), len(sparse_timestamps)

        max_len = date_len if date_len < self.max_len else self.max_len

        lengths = np.random.randint(self.min_len, max_len + 1, size=self.split_count)
        lambdas = np.random.uniform(self.llambda, self.rlambda, size=self.split_count)

        n_dense = np.floor(lengths * lambdas).astype(int)
        n_sparse = np.ceil(lengths * (1 - lambdas)).astype(int)

        idxs = [
            list(
                np.random.choice(
                    dense_timestamps, size=min(n_d, l_dense), replace=False
                )
            )
            + list(
                np.random.choice(
                    sparse_timestamps, size=min(n_s, l_sparse), replace=False
                )
            )
            for (n_d, n_s) in list(zip(n_dense, n_sparse))
        ]

        return [sorted(idx) for idx in idxs]


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
        """Overrided initialize method, which is suitable for our tasks.

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
