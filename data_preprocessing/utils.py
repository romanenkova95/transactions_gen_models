import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_features(data, mcc, client_id, date, date_format=None):
    '''
    data - pd.DataFrame with transactions data
    mcc : mcc-column-name, 
    client_id : client-id-column-name, 
    '''
    data = data.rename(columns={mcc : 'mcc', client_id: 'client_id'})

    enc = LabelEncoder()
    data['mcc'] = enc.fit_transform(data['mcc'])

    df = data.groupby('client_id')['mcc'].agg(lambda x: list(x))
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)
    
    return df

def preprocess_targets(data, client_id, target, *other_names):
    '''
    data - pd.DataFrame with targets data 
    client_id : client-id-column-name, 
    target: target-column-name
    others: names other columns that you want to keep in the target dataframe
    '''
    data = data.rename(columns={client_id : 'client_id', target: 'target'})
    
    df = data[['client_id', 'target', *other_names]].drop_duplicates()
    df.reset_index(inplace=True, drop=True)

    return df

def merge(features_df, targets_df, on='client_id'):
    '''
    Merges targets df into features df on the value 'client_id' 
    '''
    assert len(features_df) == len(targets_df), 'Dataframes do not match'
    return features_df.merge(targets_df, on=on, how='inner')
