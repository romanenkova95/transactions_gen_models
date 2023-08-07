import typing

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def by_mcc_percentiles(
    data: pd.DataFrame,
    mcc_column: str = 'mcc_code',
    transaction_amt_column: str = 'transaction_amt',
    binary_income_column: typing.Optional[str] = 'is_income',
    percentile: float = .99
) -> None:
    for mcc in tqdm(data[mcc_column].unique()):
        if binary_income_column:
            pos_pos = data[
                (data[mcc_column] == mcc) & (data[binary_income_column] == 1)
            ][transaction_amt_column].quantile(percentile)
            pos_neg = data[
                (data[mcc_column] == mcc) & (data[binary_income_column] == 0)
            ][transaction_amt_column].quantile(percentile)
        else:
            pos_pos = data[(data[mcc_column] == mcc)][transaction_amt_column].quantile(percentile)
            pos_neg = 0
        
        mcc_anomaly_target = data[data[mcc_column] == mcc][[transaction_amt_column, binary_income_column]].apply(
            lambda x: int((x[0] > pos_pos and x[1] == 1) or (x[0] > pos_neg and x[1] == 0)),
            axis=1
        )
        data.loc[mcc_anomaly_target.index, 'target'] = mcc_anomaly_target.astype(np.int32)