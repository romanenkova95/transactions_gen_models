from typing import Dict, Any

import numpy as np
import torch

from ptls.data_load.padded_batch import PaddedBatch


def is_seq_feature(k: str, v: Any) -> bool:
    if k == "event_time":
        return True
    if k.startswith("target"):
        return False
    if type(v) in (np.ndarray, torch.Tensor):
        return True 
    return False


def sliding_window_sampler(
    padded_batch: PaddedBatch,
    seq_len: int,
    stride: int = 1,
    time_col: str = "event_time",
) -> Dict[str, torch.Tensor]:
    """Sample sliding windows from the raw dataset on-the-fly.

    Args:
        padded_batch (PaddedBatch) - batch with raw input data in ptls format
        seq_len (int) - size of the sliding widow
        stride (int) - stride for sliding window procedure
        time_col (str) - name of the column containg timestamps

    Returns:
        a dictionary of the form (feature_name, feature_values)
    """
    times_batch = padded_batch.payload[time_col]
    bs, date_len = times_batch.shape
    date_range = np.arange(date_len)

    # starting positions for all the windows
    start_pos = date_range[0 : date_len - seq_len + 1 : stride]

    # list of lists of indices of the windows elements
    idxs_list = [date_range[s : s + seq_len] for s in start_pos]

    splits = {
        k: torch.stack([v[:, ix] for ix in idxs_list])
        .transpose(0, 1)
        .reshape(-1, seq_len)
        for k, v in padded_batch.payload.items()
        if is_seq_feature(k, v)
    }

    # convert into PaddedBatch format
    lengths = torch.ones(len(idxs_list) * bs, dtype=torch.int, device=times_batch.device) * seq_len
    collated_batch = PaddedBatch(splits, lengths)

    return collated_batch
