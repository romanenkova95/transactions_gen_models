import pandas as pd
from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


class DropLargeAmounts(ColTransformerPandasMixin, ColTransformer):
    """Drop large amounts.
    """
    def __init__(self, col_name_original: str, q: float):
        """Initialize DropRareMccs transform

        Args:
            col_name_original (str): original column name
            q (float): amount of mccs to keep (excluding padding token)
        """
        super().__init__(col_name_original)
        self.col_name_original = col_name_original
        self.q = q
    
    def transform(self, x: pd.DataFrame):
        quantile = x[self.col_name_original].quantile(self.q)
        return x[x[self.col_name_original] < quantile]
