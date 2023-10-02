import pandas as pd
from ptls.preprocessing.base import ColTransformer


class ToType(ColTransformer):
    """Cast certain columns to given types
    """
    def __init__(self, type_mapping: dict[str, str]):
        """Initialize transformer object.

        Args:
            type_mapping (dict[str, str]): dictionary with column names as keys, 
            and desired column types as values
        """
        self.type_mapping = type_mapping
        
    def check_is_col_exists(self, x: pd.DataFrame):
        cols_transformed = set(self.type_mapping.keys())
        cols_present = set(x.columns)
        cols_missing = cols_transformed - cols_present
        if cols_missing:
            raise ValueError(f"Missing some columns! {cols_missing=}")
    
    def transform(self, x: pd.DataFrame):
        for col_name, col_type in self.type_mapping.items():
            x[col_name] = x[col_name].astype(col_type) # type: ignore
            
        return x
