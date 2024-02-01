"""Based on code from the EasyTPP repository: https://github.com/ant-research/EasyTemporalPointProcess."""

from typing import Tuple

import torch

from ptls.data_load.padded_batch import PaddedBatch


def make_type_mask_for_pad_sequence(pad_seqs: torch.Tensor, num_event_types: int) -> torch.Tensor:
    """Make the type mask.

    Args:
    ----
        pad_seqs (tensor): a list of sequence events with equal length (i.e., padded sequence)
        num_event_types (int): number of event types in the dataset

    Returns:
    -------
        torch.Tensor: a 3-dim matrix, where the last dim (one-hot vector) indicates the type of event
    """
    type_mask = torch.zeros([*pad_seqs.shape, num_event_types], dtype=torch.int32)

    for i in range(1, num_event_types):
        type_mask[:, :, i] = pad_seqs == i

    return type_mask.to(pad_seqs.device)
    
def make_attn_mask_for_pad_sequence(pad_seqs: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Make the attention masks for the sequence.

    Args:
    ----
        pad_seqs (tensor): list of sequences that have been padded with fixed length
        pad_token_id (int): a value that used to pad the sequences

    Returns:
    -------
        torch.Tensor: a bool matrix of the same size of input, denoting the masks of the sequence (True: non mask, False: mask)


    Example:
    ```python
    seqs = [[ 1,  6,  0,  7, 12, 12],
    [ 1,  0,  5,  1, 10,  9]]
    make_attn_mask_for_pad_sequence(seqs, pad_index=12)
    >>>
        batch_non_pad_mask
        ([[ True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True,  True]])
        attention_mask
        [[[ True  True  True  True  True  True]
            [False  True  True  True  True  True]
            [False False  True  True  True  True]
            [False False False  True  True  True]
            [False False False False  True  True]
            [False False False False  True  True]]

            [[ True  True  True  True  True  True]
            [False  True  True  True  True  True]
            [False False  True  True  True  True]
            [False False False  True  True  True]
            [False False False False  True  True]
            [False False False False False  True]]]
    ```
    """
    seq_num, seq_len = pad_seqs.shape

    # [batch_size, seq_len]
    seq_pad_mask = pad_seqs == pad_token_id

    # [batch_size, seq_len, seq_len]
    attention_key_pad_mask = torch.tile(seq_pad_mask[:, None, :], (1, seq_len, 1))

    subsequent_mask = torch.tile(
        torch.triu(torch.ones((seq_len, seq_len), dtype=bool), diagonal=0)[
            None, :, :
        ],
        (seq_num, 1, 1),
    ).to(pad_seqs.device)

    attention_mask = subsequent_mask | attention_key_pad_mask

    return attention_mask
    
def restruct_batch(
    ptls_batch: PaddedBatch, col_time: str, col_type: str, pad_token_id: int, num_types: int
) -> Tuple[torch.Tensor]:
    """Restruct batch from pytorch-lifestream format (PaddedBatch with dicts of tensors) to EasyTPP format.

    Args:
    ----
        ptls_batch (PaddedBatch): padded batch in ptls format
        col_time (str): name of the column containing event times
        col_type (str): name of the column containing event types
        pad_token_id (int): a value that used to pad the sequences
        num_types (int): number of event types in the dataset

    Returns:
    ------
        A tuple of torch.Tensors:
            * time_delta - event times for a batch
            * event_types - event types for a batch
            * non_pad_mask - boolean mask indicating non-padding events
            * attention_mask - boolean mask for masked attention computation
            * type_mask - a 3-dim matrix, where the last dim (one-hot vector) indicates the type of event 
    """
    event_times = ptls_batch.payload[col_time].float()
    event_types = ptls_batch.payload[col_type]

    time_delta = event_times[:, 1:] - event_times[:, :-1]
    time_delta = torch.nn.functional.pad(time_delta, (1, 0))  # EasyTPP format

    non_pad_mask = event_times.ne(0)

    event_times[~non_pad_mask] = pad_token_id
    event_types[~non_pad_mask] = pad_token_id
    time_delta[~non_pad_mask] = pad_token_id

    type_mask = make_type_mask_for_pad_sequence(
        event_types, num_event_types=num_types
    )
    attention_mask = make_attn_mask_for_pad_sequence(
        event_types, pad_token_id=pad_token_id
    )

    return (
        event_times,
        time_delta,
        event_types,
        non_pad_mask.type(torch.int32),
        attention_mask,
        type_mask,
    )