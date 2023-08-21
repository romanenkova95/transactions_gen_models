import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.preprocessing import LabelEncoder

import holidays

RU_HOLIDAYS = holidays.country_holidays('RU')
SECONDS_IN_MONTH = 60 * 60 * 24 * 30

def preprocess_features(
    data: pd.DataFrame,
    mcc: str,
    client_id: str,
    date: str,
    amount: str,
    date_format: str = None,
    churn_horizon_months: float = None
) -> pd.DataFrame:
    """Preprocess features of the transaction dataset.
        
        :param data - pd.DataFrame with transactions data
        :param  mcc : mcc column name,
        :param  client_id : client id column name,
        :param  date : datetime column name,
        :param  amount : amount column name,
        :param  date_format - datetime format that is used in the dataset (for example, 26OCT17:00:00:00 -'%d%b%y:%H:%M:%S');
                              if None, date is the number of the day in chronological order, starting from the specified date
        :param churn_horizon_months - number of months before last transaction to be considered as 'pre-churn' period
        :return dataframe with the preprocessed features
    """
    # rename columns for consistency across different datasets
    data = data.rename(
        columns={mcc: "mcc", client_id: "client_id", date: "datetime", amount: "amount"}
    )

    # fix datetime format
    if date_format is not None:
        data["datetime"] = data["datetime"].apply(
            lambda x: pd.to_datetime(x, format=date_format)
        )
    
    # add holiday targets and weekend targets (5 - Saturday, 6 - Sunday)
    data["holiday_target"] = data["datetime"].apply(lambda x: int(x in RU_HOLIDAYS))
    data["weekend_target"] = data["datetime"].apply(lambda x: int(x.weekday() in [5, 6]))

    # encode MCC codes
    enc = LabelEncoder()
    data["mcc"] = enc.fit_transform(data["mcc"])

    df = pd.DataFrame(
        data.groupby("client_id")[["datetime", "mcc", "amount", "holiday_target", "weekend_target"]]
        .aggregate(lambda x: list(x))
    ) 
    
    # sort feature lists according to datetime
    def permute_list(original_list, perm):
        return [original_list[p] for p in perm]
    
    df["mcc"]            = df.apply(lambda x: permute_list(x.mcc, np.argsort(x.datetime)), axis=1)
    df["amount"]         = df.apply(lambda x: permute_list(x.amount, np.argsort(x.datetime)), axis=1)
    df["holiday_target"] = df.apply(lambda x: permute_list(x.holiday_target, np.argsort(x.datetime)), axis=1)
    df["weekend_target"] = df.apply(lambda x: permute_list(x.weekend_target, np.argsort(x.datetime)), axis=1)
    df["datetime"]       = df.apply(lambda x: permute_list(x.datetime, np.argsort(x.datetime)), axis=1)
    
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)
    
    # prepare 'churn' target for the Churn dataset
    if churn_horizon_months is not None:
        df["churn_target"] = df["datetime"].apply(
            lambda x: (
                np.array(list(map(lambda y: y.total_seconds(), (x[-1] - np.array(x))))) < SECONDS_IN_MONTH
            ).astype(np.int32).tolist()
        )
    
    return df

def preprocess_targets(data: pd.DataFrame, client_id: str, target: str, *other_names) -> pd.DataFrame:
    """Prepare targets for transactions dataset.
    
        :param data - pd.DataFrame with targets data
        :param client_id : client-id-column-name,
        :param target: target-column-name
        :param others: names other columns that you want to keep in the target dataframe
        :return dataframe with the preprocessed targets
    """
    # use column name "global_target" for the original labels
    data = data.rename(columns={client_id: "client_id", target: "global_target"})

    df = data[["client_id", "global_target", *other_names]].drop_duplicates()
    df.reset_index(inplace=True, drop=True)

    return df

def merge(features_df: pd.DataFrame, targets_df: pd.DataFrame, on: str = "client_id") -> pd.DataFrame:
    """Merges targets df into features df on the value 'client_id'.
    
        :param features_df - dataframe with preprocessed features
        :param targets_df - dataframe with preprocessed targets
        :param on - column name for the dataframes to be merged on
        :return result dataframe
    """
    assert len(features_df) == len(targets_df), "Dataframes do not match"
    return features_df.merge(targets_df, on=on, how="inner")

def save_parquet(res_df: pd.DataFrame, path_to_folder: str, dataset_name: str) -> None:
    """Save prepared dataset to a .parquet file.
    
        :param res_df - result dataframe to be saved
        :param path_to_folder - path to the folder with preprocessed datasets
        :param dataset_name - name of file to be used for saving
    """
    pq_table = pa.Table.from_pandas(res_df)
    pq.write_table(pq_table, path_to_folder + "/" + dataset_name + ".parquet")