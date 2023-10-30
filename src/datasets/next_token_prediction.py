"""Next token prediction dataset"""
from functools import reduce
from operator import iadd
from typing import Any, Optional

import torch

from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict

from .simple_ds import SimpleTRXDataset


class LastTokenTarget(FeatureDict):
    def __init__(self, target_seq_col: str):
        super().__init__()
        self.target_seq_col = target_seq_col

    def __call__(self, x: dict):
        seq_len = self.get_seq_len(x)
        target = x[self.target_seq_col][-1]
        new_x = self.seq_indexing(x, slice(seq_len - 1))
        return new_x, target


class NextTokenPredictionDataset(SimpleTRXDataset):
    """Dataset for next token prediction task.
    Returns tuple[last element from target column, batch]
    """

    def __init__(
        self,
        data: Any,
        min_len: int,
        random_min_seq_len: int,
        random_max_seq_len: int,
        randomize: bool = True,
        target_seq_col: str = "mcc_code",
    ):
        """Initialize dataset

        Args:
            data (Any):
                Data, compatible with ptls.datasets
            min_len (int):
                minimum sequence length (anything longer than this is filtered out)
            random_min_seq_len (int):
                minimum len of sampled subsequence
            random_max_seq_len (int):
                maximum len of sampled subsequence
            randomize (bool, optional):
                whether to use subsequence sampling. Defaults to True.
                If False, deterministically trim all sequences to min_len, leaving sequence tail.
                Used for debugging purposes (e.g. with overfit_batches).
            target_col (str):
                Name of target sequence column.
        """
        target_transform = LastTokenTarget(target_seq_col)
        super().__init__(
            data,
            min_len,
            random_min_seq_len,
            random_max_seq_len,
            randomize,
            [target_transform],
        )

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        return collate_feature_dict(x), torch.stack(y)
