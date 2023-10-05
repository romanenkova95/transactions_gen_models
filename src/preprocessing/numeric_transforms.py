from typing import Optional
import pandas as pd
from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class DropLarge(ColTransformerPandasMixin, ColTransformer):
    """Drop all values, larger than given quantile ``q``."""

    def __init__(self, col_name_original: str, q: float):
        """Initialize DropLarge transform

        Args:
            col_name_original (str): original column name
            q (float): drop all values larger than this quantile.
        """
        super().__init__(col_name_original)
        self.col_name_original = col_name_original
        self.q = q

    def transform(self, x: pd.DataFrame):
        quantile = x[self.col_name_original].quantile(self.q)
        return x[x[self.col_name_original] < quantile]


class ToType(ColTransformerPandasMixin, ColTransformer):
    """Cast certain columns to given types"""

    def __init__(
        self,
        target_type: str,
        col_name_original: str,
        col_name_target: Optional[str] = None,
        is_drop_original_col: bool = True,
    ):
        """Initialize transformer object.

        Args:
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
        col: pd.Series = x[self.col_name_original]
        col = col.astype(self.target_type)  # type: ignore
        col = col.rename(self.col_name_target)
        x = self.attach_column(x, col)
        x = super().transform(x)

        return x
