from typing import Literal, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


def time_normalization(x: pd.Series, min_timestamp: int) -> pd.Series:
    """Convert Unix timestmaps to fractions of days, shift times in the dataset.

    Args:
        x (pd.Series) - input datetime column
        min_timestamp (int) - minimum datetime in the dataframe (in Unix timestamp format)

    Returns:
        pd.Series with normalized timestamps (fraction of days)
    """
    return (
        pd.to_datetime(x).astype("datetime64[s]").astype("int64") / 1000000000
        - min_timestamp
    ) / (
        60 * 60 * 24
    )  # seconds in day


class CustomDatetimeNormalization(ColTransformerPandasMixin, ColTransformer):
    """Converts datetime column fraction of days since the earliest transaction in the dataframe.

    Args:
        col_name_original (str) - source column name
        is_drop_original_col (bool) - when target and original columns are different manage original col deletion.
    """

    def __init__(
        self,
        col_name_original: str,
        col_name_target: Optional[str] = None,
        is_drop_original_col: bool = True,
    ) -> None:
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,  # type: ignore
            is_drop_original_col=is_drop_original_col,
        )

    def fit(self, x: pd.DataFrame) -> "CustomDatetimeNormalization":
        """Record minimum timestamp"""
        super().fit(x)
        self.min_timestamp = int(x[self.col_name_original].min().timestamp())
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime column. Convert unix timestamps into days (float) since 'min_timestamp'."""
        x = self.attach_column(
            x,
            time_normalization(x[self.col_name_original], self.min_timestamp).rename(
                self.col_name_target
            ),
        )
        x = super().transform(x)
        return x


class DropDuplicates(BaseEstimator, TransformerMixin):
    """Drop duplicate rows (with same index_cols),
    keeping according to ```keep``` parameter ("first"/"last"/False).
    """

    def __init__(
        self, index_cols: list[str], keep: Literal["first", "last", False]
    ) -> None:
        """Initialize DropDuplicates transform

        Args:
            index_cols (list[str]):
                which columns to consider
            keep (Literal[&quot;first&quot;, &quot;last&quot;, False]):
                which rows to keep:
                - first - keep first row in duplicates
                - last - keep last row in duplicates
                - False - remove all duplicates
        """
        self.index_cols = index_cols
        self.keep = keep

    def fit(self, x: pd.DataFrame) -> "DropDuplicates":
        if not set(self.index_cols) < set(x.columns):
            raise ValueError(
                f"Columns mismatch! {self.index_cols} is not subset of {x.columns}"
            )

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop_duplicates(subset=self.index_cols, keep=self.keep)
