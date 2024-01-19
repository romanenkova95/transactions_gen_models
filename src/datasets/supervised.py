from ptls.frames.supervised import SeqToTargetDataset
import torch
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter

from ptls.data_load.augmentations import RandomSlice
from ptls.data_load.utils import collate_feature_dict
import numpy as np

from ptls.data_load.feature_dict import FeatureDict


class CustomSupervisedDataset(SeqToTargetDataset):
    """Custom supervised dataset inherited from ptls supervised ds."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int,
        max_len: int = 1000,
        target_col_name: str = "global_target",
        *args,
        **kwargs
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimum subsequence length
            max_len (int): maximum subsequence length
            target_col_name (str, optional): column name of global target
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            target_col_name,
            target_dtype=torch.int64,
        )
        self.crop = RandomSlice(min_len, max_len)

    def __getitem__(self, item):
        elem = self.data[item]
        return self.crop(elem)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.crop(feature_arrays)


class CustomSupervisedNoSplitDataset(SeqToTargetDataset):
    """Custom supervised dataset inherited from ptls supervised ds."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int = 15,
        target_col_name: str = "global_target",
        *args,
        **kwargs
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            target_col_name (str, optional): column name of global target
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            target_col_name,
            target_dtype=torch.int64,
        )


class CustomSupervisedLocalTargetDataset(CustomSupervisedDataset):
    """Custom supervised dataset inherited from ptls supervised ds."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int = 15,
        max_len: int = 150,
        target_col_name: str = "local_target",
        *args,
        **kwargs
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            target_col_name (str, optional): column name of global target
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            min_len,
            max_len,
            target_col_name,
            target_dtype=torch.int64,
        )

    def collate_fn(self, padded_batch):
        padded_batch = collate_feature_dict(padded_batch)
        target = padded_batch.payload[self.target_col_name][:, -1]
        del padded_batch.payload[self.target_col_name]
        if self.target_dtype is not None:
            target = target.to(dtype=self.target_dtype)
        return padded_batch, target


class SliceLastN(FeatureDict):
    """
    This class is used as 'f_augmentation' argument for
    ptls.data_load.datasets.augmentation_dataset.AugmentationDataset (AugmentationIterableDataset).
    """

    def __init__(self, n: int = 50):
        super().__init__()

        self.n = n

    def __call__(self, x):
        seq_len = self.get_seq_len(x)
        idx = self.get_idx(seq_len)
        new_x = self.seq_indexing(x, idx)
        return new_x

    def get_idx(self, seq_len):
        new_idx = np.arange(seq_len)
        return new_idx[-self.n :]


class CustomLastNSupervisedDataset(SeqToTargetDataset):
    """Custom supervised dataset inherited from ptls supervised ds."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        min_len: int,
        n: int = 50,
        target_col_name: str = "global_target",
        *args,
        **kwargs
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimum subsequence length
            max_len (int): maximum subsequence length
            target_col_name (str, optional): column name of global target
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            target_col_name,
            target_dtype=torch.int64,
        )
        self.crop = SliceLastN(n)

    def __getitem__(self, item):
        elem = self.data[item]
        return self.crop(elem)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.crop(feature_arrays)
