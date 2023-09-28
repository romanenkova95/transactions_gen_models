from typing import Dict
import pandas as pd
from ptls.preprocessing.base import ColTransformer


class ToType(ColTransformer):
    def __init__(self, type_mapping: Dict[str, str]):
        self.type_mapping = type_mapping
        
    def fit(self, x):
        return self
    
    def transform(self, x: pd.DataFrame):
        for col_name, col_type in self.type_mapping.items():
            x[col_name] = x[col_name].astype(col_type) # type: ignore
            
        return x
