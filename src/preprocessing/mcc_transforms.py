import pandas as pd
from ptls.preprocessing.base import ColTransformer


class DropLargeMccs(ColTransformer):
    """Drops mccs larger than n_mccs_keep. 
    To be used after frequency encoder to drop the rarest mcc codes.
    """
    def __init__(self, col_name_original: str, n_mccs_keep: int):
        self.col_name_original = col_name_original
        self.n_mccs_keep = n_mccs_keep

    def fit(self, x):
        return self
    
    def transform(self, x: pd.DataFrame):
        x = x[x[self.col_name_original] < self.n_mccs_keep]
        return x
