"""Module for the deterministic dataset, which yields sliding window slices from users."""

from ptls.data_load import AugmentationChain
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from torch.utils.data import IterableDataset


class SlidingWindowDataset(IterableDataset, FeatureDict):
    """A simple dataset for yielding sliding-window slices from users."""
    
    def __init__(
        self,
        data: list[dict],
        window_size: int,
        window_step: int,
        f_augmentations=None,
        collate_fn=None,
    ):
        """Initialize module's internal state.

        Args:
        ----
            data (list[dict]): the data to yield from.
            window_size (int): the size of the sliding window.
            window_step (int): the step of the sliding window.
            f_augmentations (_type_, optional): 
                augmentations to apply to cropped slices before yielding. Defaults to None.
            collate_fn (_type_, optional): 
                collation function for PtlsDatamodule. Defaults to None, in which case use collate_feature_dict.
        """
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.window_step = window_step
        self.augmentations = AugmentationChain(f_augmentations)
        self.collate_fn = collate_fn or collate_feature_dict

    def __iter__(self):
        """Iterate through all users in the dataset with a sliding window.

        Yields
        ------
            dict: cropped and augmented slices from user sequences.
        """
        for row in self.data:
            seq_len = self.get_seq_len(row)
            for start in range(0, seq_len - self.window_size, self.window_step):
                end = start + self.window_size
                cropped_row = self.seq_indexing(row, slice(start, end))
                yield self.augmentations(cropped_row)
