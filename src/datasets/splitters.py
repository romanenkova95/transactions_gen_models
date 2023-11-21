from ptls.frames.coles.split_strategy import AbsSplit
import numpy as np
from typing import List


class TimeCLSampler(AbsSplit):
    """
    TimeCL sampler implementation, ptls-style.
    For details, see Algorithm 1 from the paper:
        http://mesl.ucsd.edu/pubs/Ranak_AAAI2023_PrimeNet.pdf
    Args:
        min_len (int): minimum subsequence length
        max_len (int): maximum subsequence length
        llambda (float): lower bound for lambda value
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

        max_len = date_len if date_len < self.max_len else self.max_len

        lengths = np.random.randint(self.min_len, max_len, size=self.split_count)
        lambdas = np.random.uniform(self.llambda, self.rlambda, size=self.split_count)

        n_dense, n_sparse = np.floor(lengths * lambdas).astype(int), np.ceil(
            lengths * (1 - lambdas)
        ).astype(int)

        idxs = [
            list(
                np.random.choice(
                    dense_timestamps,
                    size=min(n_d, len(dense_timestamps)),
                    replace=False,
                )
            )
            + list(
                np.random.choice(
                    sparse_timestamps,
                    size=min(n_s, len(sparse_timestamps)),
                    replace=False,
                )
            )
            for (n_d, n_s) in list(zip(n_dense, n_sparse))
        ]

        return [sorted(idx) for idx in idxs]


class SampleLength(AbsSplit):
    """
    A sampler that cuts only first N transactions.
    """

    def __init__(self, length):
        self.length = length

    def split(self, dates):
        return [np.arange(self.length)]
