import pandas as pd
from ptls.preprocessing.pandas.frequency_encoder import FrequencyEncoder


class DropRare(FrequencyEncoder):
    """Encode categories frequency-wise, and keep only top-k most frequent categories. 
    To be used after frequency encoder to drop the rarest mcc codes.
    """
    def __init__(self, col_name_original: str, k: int):
        """Initialize DropRare transform

        Args:
            col_name_original (str): original column name
            k (int): amount of categories to keep (excluding padding token)
        """
        super().__init__(col_name_original)
        self.col_name_original = col_name_original
        self.k = k
    
    def transform(self, x: pd.DataFrame):
        x = super().transform(x)
        return x[x[self.col_name_original] < self.k]
