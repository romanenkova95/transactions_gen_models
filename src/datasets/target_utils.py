"""Module with utils for creating datasets with targets."""

from typing import Any

import torch
from ptls.data_load import PaddedBatch
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from torch import Tensor


def collate_fn_with_targets(batch: list[tuple]) -> tuple[PaddedBatch, Tensor]:
    """Collation function for batch with targets.

    Args:
    ----
        batch: tuples of data and targets

    Returns:
    -------
        tuple:
            tuple of two tensors: first contains PaddedaBatch of data, the second contains targets.
    """
    x, y = zip(*batch)
    return collate_feature_dict(x), torch.stack(y)


class LastTokenTarget(FeatureDict):
    """Dataset transform to set the target as the last token of a sequence."""

    def __init__(self, target_seq_col: str, drop_last: bool):
        """Initialize internal module state.

        Args:
        ----
            target_seq_col (str):
                the column from which to take the last element when creating the target.
            drop_last (bool):
                whether to drop the target element, or to pass it along with the rest of the data.
        """
        super().__init__()
        self.target_seq_col = target_seq_col
        self.drop_last = drop_last

    def __call__(self, x: dict):
        """Apply transform.

        Args:
        ----
            x (dict): data dict

        Returns:
        -------
            tuple: data dict (optionally cropped, if drop_last) and target.
        """
        target = x[self.target_seq_col][-1]
        if self.drop_last:
            seq_len = self.get_seq_len(x)
            x = self.seq_indexing(x, slice(seq_len - 1))
        return x, target


class TimeDiffTarget(LastTokenTarget):
    """Transform to set the target as the distance in time to the last transaction and drop the last transaction."""

    def __init__(self, time_col: str = "event_time"):
        """Initialize internal module state.

        Args:
        ----
            time_col (str, optional): the name of the time column. Defaults to "event_time".
        """
        super().__init__(time_col, True)

    def __call__(self, x: dict) -> Any:
        """Apply the transformation.

        Args:
        ----
            x (dict): the dict of data

        Returns:
        -------
            tuple: the data (cropped) and the target (distance in time from pre-last to last transaction).
        """
        x, next_timestamp = super().__call__(x)
        time_diff = next_timestamp - x[self.target_seq_col][-1]
        return x, time_diff
