"""File with tools for data preprocessing."""
import random
from pathlib import Path

import pandas as pd
from ptls.data_load.datasets import MemoryMapDataset
from ptls.preprocessing import PandasDataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_PTH = Path("/home/COLES/poison/data")


def split_dataset(
    df_trx, df_target, test_size: float = 0.2, val_size: float = 0.1, seed: int = 142
):
    """Split the transactional data into train, test and validation.

    Args:
    ----
        df_trx: pd.DataFrame - transactional data
        df_target: pd.DataFrame - target data
        test_size: float - the proportion of the dataset to include in the test split
        val_size: float - the proportion of the dataset to include in the val split
        seed: int - random state of the split

    Returns:
    -------
        (pd.DataFrame, pd.DataFrame, pd.DataFrame) - train, valid and test dataframes
    """
    test_size, val_size = (
        int(len(df_target) * test_size),
        int(len(df_target) * val_size),
    )

    df_target_train, df_target_test = train_test_split(
        df_target, test_size=test_size, stratify=df_target["target"], random_state=seed
    )
    df_target_train, df_target_valid = train_test_split(
        df_target_train,
        test_size=val_size,
        stratify=df_target_train["target"],
        random_state=seed,
    )
    print(
        "Split {} records to train: {}, valid: {}, test: {}".format(
            *[
                len(df)
                for df in [df_target, df_target_train, df_target_valid, df_target_test]
            ]
        )
    )

    df_trx_train = pd.merge(df_trx, df_target_train["uid"], on="uid", how="inner")
    df_trx_valid = pd.merge(df_trx, df_target_valid["uid"], on="uid", how="inner")
    df_trx_test = pd.merge(df_trx, df_target_test["uid"], on="uid", how="inner")
    print(
        "Split {} transactions to train: {}, valid: {}, test: {}".format(
            *[len(df) for df in [df_trx, df_trx_train, df_trx_valid, df_trx_test]]
        )
    )

    return df_trx_train, df_trx_valid, df_trx_test


def df_to_dataset(
    df_trx_train,
    df_trx_valid,
    df_trx_test,
    df_target,
    event_time_transformation="dt_to_timestamp",
):
    """Merge target data with train, val and test transactions data.

    Transforms dataframes into MemoryMapDataset objects.

    Args:
    ----
        df_trx_train: pd.DataFrame - train dataset with transactional data
        df_trx_valid: pd.DataFrame - validation dataset with transactional data
        df_trx_test: pd.DataFrame - test dataset with transactional data
        df_target: pd.DataFrame - dataset with targets
        event_time_transformation: str - the kind of transformation
            that is applied to the datetime column.

    Returns:
    -------
        (MemoryMapDataset, MemoryMapDataset, MemoryMapDataset):
            train dataset, validation dataset and test dataset
    """
    preprocessor = PandasDataPreprocessor(
        col_id="uid",
        col_event_time="trans_dttm",
        event_time_transformation=event_time_transformation,
        cols_category=["category"],
        cols_numerical=["amount"],
        return_records=False,
    )

    df_data_train = preprocessor.fit_transform(df_trx_train)
    df_data_valid = preprocessor.transform(df_trx_valid)
    df_data_test = preprocessor.transform(df_trx_test)

    df_data_train = pd.merge(df_data_train, df_target, on="uid")  # type: ignore
    df_data_valid = pd.merge(df_data_valid, df_target, on="uid")
    df_data_test = pd.merge(df_data_test, df_target, on="uid")

    df_data_train = df_data_train.to_dict(orient="records")
    df_data_valid = df_data_valid.to_dict(orient="records")
    df_data_test = df_data_test.to_dict(orient="records")

    dataset_train = MemoryMapDataset(df_data_train)
    dataset_valid = MemoryMapDataset(df_data_valid)
    dataset_test = MemoryMapDataset(df_data_test)

    return dataset_train, dataset_valid, dataset_test


def load_dataset(ds_name: str):
    """Load the data.

    Args:
    ----
        ds_name: str - "age", "default", "churn" or "raif"

    Returns:
    -------
        (MemoryMapDataset, MemoryMapDataset, MemoryMapDataset):
            train dataset, validation dataset and test dataset
    """
    if ds_name == "age":
        df_target = pd.read_csv(BASE_PTH / "age" / "train_target.csv")
        df_target = df_target.rename(
            columns={
                "bins": "target",
                "client_id": "uid",
            }
        )

        df_trx = pd.read_csv(BASE_PTH / "age" / "transactions_train.csv")
        df_trx = df_trx.rename(
            columns={
                "client_id": "uid",
                "trans_date": "trans_dttm",
                "small_group": "category",
                "amount_rur": "amount",
            }
        )

    elif ds_name == "default":
        df_target = pd.read_csv(BASE_PTH / "default" / "target_finetune.csv")
        df_target = df_target.rename(
            columns={
                "user_id": "uid",
            }
        )

        df_trx = pd.read_csv(BASE_PTH / "default" / "transactions_finetune.csv")
        df_trx = df_trx.rename(
            columns={
                "user_id": "uid",
                "transaction_dttm": "trans_dttm",
                "mcc_code": "category",
                "transaction_amt": "amount",
            }
        )

    elif ds_name == "churn":
        df = pd.read_csv(BASE_PTH / "churn" / "train.csv")
        df = df[df["currency"] == 810]
        df = df.rename(
            columns={
                "cl_id": "uid",
                "target_flag": "target",
                "TRDATETIME": "trans_dttm",
                "MCC": "category",
            }
        )
        enc = LabelEncoder()
        df["category"] = enc.fit_transform(df["category"])
        df["trans_dttm"] = pd.to_datetime(df["trans_dttm"], format="%d%b%y:%H:%M:%S")

        df_target = df[["uid", "target"]].drop_duplicates()
        df_target.reset_index(inplace=True, drop=True)

        df_trx = df[["uid", "category", "trans_dttm", "amount"]]

    elif ds_name == "raif":
        df_target = pd.read_csv(
            BASE_PTH / "raif" / "clients_last_2_fixed.csv", delimiter=";"
        )

        married_cnums = list(df_target["cnum_"][df_target["married_"] == "married"])
        not_married_cnums = list(
            df_target["cnum_"][df_target["married_"] == "not_married"]
        )
        rand_clients = random.sample(married_cnums, 432254) + random.sample(
            not_married_cnums, 823671
        )

        df_target.drop(
            df_target[df_target["cnum_"].isin(rand_clients)].index, inplace=True
        )

        df_target = df_target.rename(columns={"cnum_": "uid", "married_": "target"})
        df_target["target"] = df_target["target"].map({"not_married": 0, "married": 1})

        df_trx = pd.read_csv(
            BASE_PTH / "raif" / "transactions_last_2.csv", delimiter=";"
        )
        df_trx = df_trx.drop(columns=["mrchcity", "mrchname"])
        df_trx = df_trx.rename(
            columns={
                "cnum": "uid",
                "purchdate": "trans_dttm",
                "mcc": "category",
            }
        )

    df_trx_train, df_trx_valid, df_trx_test = split_dataset(df_trx, df_target)  # type: ignore

    dttm_transformation = "none" if ds_name == "age" else "dt_to_timestamp"

    return df_to_dataset(
        df_trx_train,
        df_trx_valid,
        df_trx_test,
        df_target,  # type: ignore
        event_time_transformation=dttm_transformation,
    )
