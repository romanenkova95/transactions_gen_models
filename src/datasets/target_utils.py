from typing import Any
from ptls.data_load.feature_dict import FeatureDict

import torch

from ptls.data_load.utils import collate_feature_dict


def collate_fn_with_targets(batch):
    x, y = zip(*batch)
    return collate_feature_dict(x), torch.stack(y)


class LastTokenTarget(FeatureDict):
    def __init__(self, target_seq_col: str, drop_last: bool):
        super().__init__()
        self.target_seq_col = target_seq_col
        self.drop_last = drop_last

    def __call__(self, x: dict):
        target = x[self.target_seq_col][-1]
        if self.drop_last:
            seq_len = self.get_seq_len(x)
            x = self.seq_indexing(x, slice(seq_len - 1))
        return x, target


class TimeDiffTarget(LastTokenTarget):
    def __init__(self, time_col: str = "event_time"):
        super().__init__(time_col, True)

    def __call__(self, x: dict) -> Any:
        x, next_timestamp = super().__call__(x)
        time_diff = next_timestamp - x[self.target_seq_col][-1]
        return x, time_diff
