from typing import Any
from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.augmentations import RandomSlice, SeqLenLimit
from ptls.data_load.utils import collate_feature_dict


class SimpleTRXDataset(AugmentationDataset):
    def __init__(
        self, 
        data: Any, 
        min_len: int, 
        random_min_seq_len: int, 
        random_max_seq_len: int,
        randomize: bool = True
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
        """
        augmentations = []
        if randomize:
            augmentations.append(RandomSlice(random_min_seq_len, random_max_seq_len))
        else:
            augmentations.append(SeqLenLimit(int((random_min_seq_len + random_max_seq_len) / 2), strategy="head"))
        
        super().__init__(
            MemoryMapDataset(data, [SeqLenFilter(min_len)]),
            augmentations
        )

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)
