"""Module for the simple dataset, which randomly crops user sequences."""
from typing import Any, Callable, Optional

from ptls.data_load.augmentations import RandomSlice
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.utils import collate_feature_dict


class RandomCropDataset(AugmentationDataset):
    """Simple dataset, which randomly crops user sequences."""

    def __init__(
        self,
        data: Any,
        random_min_seq_len: int,
        random_max_seq_len: int,
        f_augmentations: Optional[list] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """Initialize dataset.

        Args:
        ----
            data (Any):
                Data, compatible with ptls.datasets
            random_min_seq_len (int):
                minimum len of sampled subsequence
            random_max_seq_len (int):
                maximum len of sampled subsequence
            f_augmentations (list):
                post-slice augmentations list (callables, applied to feature dict).
                Defaults to None, for no augmentations
            collate_fn (callable):
                collate_fn argument for dataloader, defaults to None for collate_feature_dict
        """
        augmentations = [RandomSlice(random_min_seq_len, random_max_seq_len)]
        augmentations.extend(f_augmentations or [])
        super().__init__(data, augmentations)
        self.collate_fn = collate_fn or collate_feature_dict
