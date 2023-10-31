from torch.utils.data import IterableDataset
from ptls.data_load import AugmentationChain
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict


class SlidingWindowDataset(IterableDataset, FeatureDict):
    def __init__(
        self,
        data: list[dict],
        window_size: int,
        window_step: int,
        f_augmentations=None,
        collate_fn=None,
    ):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.window_step = window_step
        self.augmentations = AugmentationChain(f_augmentations)
        self.collate_fn = collate_fn or collate_feature_dict

    def __iter__(self):
        for row in self.data:
            seq_len = self.get_seq_len(row)
            for start in range(0, seq_len - self.window_size, self.window_step):
                end = start + self.window_size
                cropped_row = self.seq_indexing(row, slice(start, end))
                yield self.augmentations(cropped_row)
