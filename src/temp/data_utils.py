from datetime import timedelta

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def split_into_samples(
    data: pd.DataFrame,
    time_delta: int = 7,
    len_min: int = 40,
    len_max: int = 120,
    user_column_name: str = 'user_id',
    data_column_name: str = 'transaction_dttm'
) -> None:
    
    label_column = np.zeros(data.shape[0])
    start_user = data.iloc[0][user_column_name]

    start_user = data.iloc[0][user_column_name]
    index = 0
    start_time = data.iloc[0][data_column_name]
    count = 0
    index_to_drop = []
    increment = 0

    for i in tqdm(range(len(data)), leave=True):
        curr_time = data.iloc[i][data_column_name]
        curr_user = data.iloc[i][user_column_name]
        count += 1
        if curr_user != start_user:
            if len_min > count < len_max:
                index_to_drop.append(index)
            count = 1
            index += 1
            start_user = curr_user
            start_time = curr_time
        elif curr_time > start_time + timedelta(days=time_delta):
            start_time = curr_time
            if count >= len_min:
                if len_min > count < len_max:
                    index_to_drop.append(index)
                count = 1
                index += 1
        label_column[i] = index
    if len_min > count < len_max:
        index_to_drop.append(index)
    
    data['sample_label'] = np.array(label_column)
    data.drop(index=data[data['sample_label'].isin(index_to_drop)].index, axis=0, inplace=True)
    data['sample_label'] = data['sample_label'].astype(np.int32)