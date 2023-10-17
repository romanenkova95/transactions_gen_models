"""
Custom coles datamodule
"""
import torch
import pandas as pd

from typing import List, Dict

from functools import reduce
from operator import iadd
from ptls.data_load.utils import collate_feature_dict

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices

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

class CustomUserValidationColesDataset(ColesDataset):

    def __init__(self,
                 *args, 
                 target_cols: List[str] = None,
                 **kwargs, ):
        super().__init__(*args, **kwargs) 

        self.target_cols = target_cols

    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        full_targets = {}
        for target_col in self.target_cols:
            full_targets[target_col] = feature_arrays[target_col]

        return [self.get_splits(feature_arrays), full_targets]
    
    def collate_targets(self, targets: List[Dict]):
        targets = pd.DataFrame(targets)
        result = {}
        for col in targets.columns:
            result[col] = torch.Tensor(targets[col].values)
        return result
    
    def collate_fn(self, batch):
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(reduce(iadd, batch[::2]))

        return padded_batch, self.collate_targets(batch[1::2]) 