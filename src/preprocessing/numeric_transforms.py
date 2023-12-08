"""Module with transforms of numeric columns."""

from typing import Optional
import pandas as pd
from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class DropQuantile(ColTransformerPandasMixin, ColTransformer):
    """Drop all values, which don't fall between ``q_min`` and ``q_max``."""

    def __init__(self, col_name_original: str, q_min: float, q_max: float):
        """Initialize DropLarge transform.

        Args:
        ----
            col_name_original (str): original column name
            q (float): drop all values larger than this quantile.
        """
        super().__init__(col_name_original)
        self.q_min = q_min
        self.q_max = q_max

    def transform(self, x: pd.DataFrame):
        """Transform the data to fall between the specified quantiles.

        Args:
        ----
            x (pd.DataFrame): the data to transform.

        Returns:
        -------
            pd.DataFrame: the transformed data, will fall between the required quantiles.
        """
        column = x[self.col_name_original]
        q_min = column.quantile(self.q_min)
        q_max = column.quantile(self.q_max)
        return x[(q_min < column) | (column < q_max)]


class ToType(ColTransformerPandasMixin, ColTransformer):
    """Cast certain columns to given types."""

    def __init__(
        self,
        target_type: str,
        col_name_original: str,
        col_name_target: Optional[str] = None,
        is_drop_original_col: bool = True,
    ):
        """Initialize transformer object.

        Args:
        ----
            target_type (str):
                desired column type
            col_name_original (str):
                name of column to be transformed
            col_name_target (str):
                name of transformed column to be appended, defaults to col_name_original
            is_drop_original (bool):
                whether to drop original column after transformation, defaults to True
        """
        super().__init__(col_name_original, col_name_target, is_drop_original_col)  # type: ignore
        self.target_type = target_type

    def transform(self, x: pd.DataFrame):
        """Transform the data, casting the selected column to the selected type.

        Args:
        ----
            x (pd.DataFrame): the data to transform.

        Returns:
        -------
            DataFrame: transformed data, with the selected column of required type.
        """
        col: pd.Series = x[self.col_name_original]
        col = col.astype(self.target_type)  # type: ignore
        col = col.rename(self.col_name_target)
        x = self.attach_column(x, col)
        return super().transform(x)
