from ptls.frames.supervised import SeqToTargetDataset
import torch


class CustomSupervisedDataset(SeqToTargetDataset):
    """Custom supervised dataset inherited from ptls supervised ds."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        target_col_name: str = "global_target",
        deterministic: bool = False,
    ):
        """Initialize internal module state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimum subsequence length
            max_len (int): maximum subsequence length
            target_col_name (str, optional): column name of global target
        """
        super().__init__(
            data,
            target_col_name,
            target_dtype=torch.int64,
        )
