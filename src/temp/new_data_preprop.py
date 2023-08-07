import os
import pickle
import typing

import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.utils.logging_utils import get_logger
from .data_utils import split_into_samples
from .anomaly_scheme import by_mcc_percentiles


logger = get_logger(name=__name__)

def preprocessing(
    cfg: DictConfig, return_preproc: bool = False
) -> typing.Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    
    dir_path: str                   = cfg['dir_path']
    ignore_existing_preproc: bool   = cfg['ignore_existing_preproc']
    drop_currency: bool             = cfg['preproc']['drop_currency']
    time_delta: int                 = cfg['preproc']['time_delta']
    len_min: int                    = cfg['preproc']['len_min']
    len_max: int                    = cfg['preproc']['len_max']
    anomaly_strategy: str           = cfg['anomaly_strategy']
    percentile: float               = cfg['percentile']

    preproc_dir_path = os.path.join(dir_path, 'preprocessed')
    if not os.path.exists(preproc_dir_path):
        logger.warning('Preprocessing folder does not exist. Creating...')
        os.mkdir(preproc_dir_path)
    if ignore_existing_preproc:
        logger.info('Preprocessing will ignore all previously saved files')

    if (
        os.path.exists(os.path.join(preproc_dir_path, 'preproc_dataset.parquet')) and \
        not ignore_existing_preproc
    ):
        data_srt = pd.read_parquet(os.path.join(preproc_dir_path, 'preproc_dataset.parquet'))
    
    else:
        df_transactions = pd.read_parquet(os.path.join(dir_path, 'transactions.parquet'))

        logger.info('Transfering timestamp to the datetime format')
        df_transactions['transaction_dttm'] = pd.to_datetime(
            df_transactions['transaction_dttm'],
            format='%Y-%m-%d %H:%M:%S'
        )
        logger.info('Done!')

        df_transactions.drop(
            index=df_transactions[df_transactions['mcc_code'] == -1].index,
            axis=0,
            inplace=True
        )

        if drop_currency:
            logger.info('Dropping currency_rk')
            df_transactions.drop(
                index=df_transactions[df_transactions['currency_rk'] != 48].index,
                axis=0,
                inplace=True
            )
            df_transactions.drop(columns=['currency_rk'], axis=1, inplace=True)
            logger.info('Done!')

        if (
            os.path.exists(os.path.join(preproc_dir_path, 'mcc2id.dict')) and \
            not ignore_existing_preproc
        ):
            with open(os.path.join(preproc_dir_path, 'mcc2id.dict'), 'rb') as f:
                mcc2id = dict(pickle.load(f))
        else:
            mcc2id = dict(zip(
                df_transactions['mcc_code'].unique(), 
                np.arange(df_transactions['mcc_code'].nunique()) + 1
            ))
            with open(os.path.join(preproc_dir_path, 'mcc2id.dict'), 'wb') as f:
                pickle.dump(mcc2id, f)
        
        df_transactions['mcc_code'] = df_transactions['mcc_code'].map(mcc2id)

        df_transactions['is_income'] = (df_transactions['transaction_amt'] > 0).astype(np.int32)
        df_transactions['transaction_amt'] = df_transactions[['transaction_amt', 'is_income']].apply(
            lambda t: np.log(t[0]) if t[1] else np.log(-t[0]),
            axis=1
        )

        if (
            os.path.exists(os.path.join(preproc_dir_path, 'user2id.dict')) and \
            not ignore_existing_preproc
        ):
            with open(os.path.join(preproc_dir_path, 'user2id.dict'), 'rb') as f:
                user2id = dict(pickle.load(f))
        else:
            user2id = dict(zip(
                df_transactions['user_id'].unique(), 
                np.arange(df_transactions['user_id'].nunique()) + 1
            ))
            with open(os.path.join(preproc_dir_path, 'user2id.dict'), 'wb') as f:
                pickle.dump(user2id, f)
        df_transactions['user_id'] = df_transactions['user_id'].map(user2id)
        
        data_srt = df_transactions.sort_values(['user_id','transaction_dttm']).reset_index(drop=True)

        logger.info('Start splitting into samples')
        split_into_samples(data_srt, time_delta, len_min, len_max)
        logger.info('Done!')
        logger.info('Anomaly splitting')
        if anomaly_strategy == 'quantile':
            by_mcc_percentiles(data_srt, percentile=percentile)
        logger.info('Done!')
        anomaly_samples = data_srt['target'].sum()
        normal_samples = data_srt.shape[0] - anomaly_samples
        logger.info(f'Normal samples count - {int(normal_samples)}. Anomaly Samples - {int(anomaly_samples)}')

        data_srt.to_parquet(os.path.join(preproc_dir_path, 'preproc_dataset.parquet'))

    if (
        os.path.exists(os.path.join(preproc_dir_path, 'agg_dataset.parquet')) and \
        not ignore_existing_preproc
    ):
        data_agg = pd.read_parquet(os.path.join(preproc_dir_path, 'agg_dataset.parquet'))
    
    else:
        data_agg = data_srt.groupby('sample_label').agg({
            'user_id': lambda x: x.iloc[0],
            'mcc_code': lambda x: x.tolist(),
            'is_income': lambda x: x.tolist(),
            'transaction_amt': lambda x: x.tolist(),
            'target': lambda x: x.sum() / x.count()
        })
        data_agg.to_parquet(os.path.join(preproc_dir_path, 'agg_dataset.parquet'))

    if return_preproc:
        return data_agg, data_srt
    else:
        return data_agg
