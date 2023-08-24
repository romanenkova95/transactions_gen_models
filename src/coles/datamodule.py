"""
Custom coles datamodule
"""
import numpy as np

from functools import reduce
from operator import iadd

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleUniform
from ptls.data_load.utils import collate_feature_dict


class AbsSplit:
    def split(self, dates):
        raise NotImplementedError()
        
class SampleAll(AbsSplit):
    def __init__(self, seq_len, stride, **_):
        self.seq_len = seq_len
        self.stride = stride

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        if date_len <= self.seq_len:
            return [date_range for _ in range(date_len)]

        # crop last 'seq_len' record as we do not have local labels for them
        start_pos = date_range[0 : date_len - self.seq_len : self.stride]
        return [date_range[s:s + self.seq_len] for s in start_pos]


class CustomColesDataset(ColesDataset):
    """
    Custom coles dataset inhereted from ptls coles dataset.
    """

    def __init__(
        self,
        data: list[dict],
        min_len: int,
        split_count: int,
        seq_len: int,
        *args,
        col_time: str = 'event_time',
        **kwargs
    ):
        """Overrided initialize method, which is suitable for our tasks

        Args:
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            split_count (int): number of splitting samples
            seq_len (int): length of the uniform sampled subsequences
            col_time (str, optional): column name with event time. Defaults to 'event_time'.
        """
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleUniform(split_count, seq_len),
            col_time,
            *args, **kwargs
        )

class CustomColesValidationDataset(ColesDataset):
    def __init__(
        self,
        data: list[dict],
        min_len: int,
        seq_len: int,
        stride: int,
        *args,
        col_time: str = 'event_time',
        local_target_col: str = None,
        **kwargs
    ):
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            SampleAll(seq_len, stride),
            col_time,
            *args, **kwargs
        )
        self.local_target_col = local_target_col

    def collate_fn(self, batch):
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        if self.local_target_col is not None:
            # for every window of size 'seq_len', take local label corresponding for the last time step in this window
            local_targets = padded_batch.payload[self.local_target_col][:, -1]
            return padded_batch, local_targets
        else:
            class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples]
            return padded_batch, torch.LongTensor(class_labels)
        