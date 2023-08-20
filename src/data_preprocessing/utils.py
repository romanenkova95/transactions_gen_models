import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_features(data, mcc, client_id, date, amount, date_format=None):
    """
    data - pd.DataFrame with transactions data
    mcc : mcc column name,
    client_id : client id column name,
    date : datetime column name,
    amount : amount column name,
    date_format - datetime format that is used in the dataset (for example, 26OCT17:00:00:00 -'%d%b%y:%H:%M:%S');
    if None, date is the number of the day in chronological order, starting from the specified date
    """
    data = data.rename(
        columns={mcc: "mcc", client_id: "client_id", date: "datetime", amount: "amount"}
    )

    if date_format is not None:
        data["datetime"] = data["datetime"].apply(
            lambda x: pd.to_datetime(x, format=date_format)
        )

    enc = LabelEncoder()
    data["mcc"] = enc.fit_transform(data["mcc"])

    df = pd.DataFrame(
        data.groupby("client_id")[["datetime", "mcc", "amount"]]
        .aggregate(list)
        .aggregate(lambda x: sorted(list(zip(*x)), key=lambda x: x[0]), axis=1)
    )
    df = df.rename(columns={0: "transactions"})
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)

    return df


def preprocess_targets(data, client_id, target, *other_names):
    """
    data - pd.DataFrame with targets data
    client_id : client-id-column-name,
    target: target-column-name
    others: names other columns that you want to keep in the target dataframe
    """
    data = data.rename(columns={client_id: "client_id", target: "target"})

    df = data[["client_id", "target", *other_names]].drop_duplicates()
    df.reset_index(inplace=True, drop=True)

    return df


def merge(features_df, targets_df, on="client_id"):
    """
    Merges targets df into features df on the value 'client_id'
    """
    assert len(features_df) == len(targets_df), "Dataframes do not match"
    return features_df.merge(targets_df, on=on, how="inner")
