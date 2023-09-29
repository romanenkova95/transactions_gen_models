import pandas as pd
from ptls.preprocessing.pandas.frequency_encoder import FrequencyEncoder


class DropRareMccs(FrequencyEncoder):
    """Encode mccs frequency-wise, and keep only top-k most frequent mccs. 
    To be used after frequency encoder to drop the rarest mcc codes.
    """
    def __init__(self, col_name_original: str, k: int):
        """Initialize DropRareMccs transform

        Args:
            col_name_original (str): original column name
            k (int): amount of mccs to keep (excluding padding token)
        """
        super().__init__(col_name_original)
        self.col_name_original = col_name_original
        self.k = k

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        return self
    
    def transform(self, x: pd.DataFrame):
        x = super().transform(x)
        return x[x[self.col_name_original] < self.k]
