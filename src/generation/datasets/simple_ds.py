from typing import Any
from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.augmentations import RandomSlice
from ptls.data_load.utils import collate_feature_dict


class SimpleTRXDataset(AugmentationDataset):
    def __init__(self, data: Any, min_len: int, random_min_seq_len: int, random_max_seq_len: int):
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            [RandomSlice(random_min_seq_len, random_max_seq_len)]
        )

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)
