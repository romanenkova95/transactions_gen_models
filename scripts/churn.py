import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
import argparse

from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.datasets import MemoryMapDataset
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import loggers as pl_loggers

import torchmetrics
from ptls.nn import TrxEncoder, RnnSeqEncoder

from functools import partial
from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
from ptls.frames import PtlsDataModule

from ptls.data_load.utils import collate_feature_dict
import random
from ptls.frames.inference_module import InferenceModule
from sklearn.metrics import f1_score, roc_auc_score

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

base_path = Path('/home/COLES/poison/data')

class RNNclassifier(torch.nn.Module):
    def __init__(self,
                input_size: int = None,
                hidden_size: int = None,
                num_layers: int = 1,
                bias: bool = True,
                batch_first: bool = True,
                dropout: float = 0,
                bidirectional: bool = False,
                num_classes: int = 1,
                logsoftmax: bool = True):

        super().__init__()

        self.backbone = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        if logsoftmax:
            activation = torch.nn.LogSoftmax(dim=-1)
        else:
            activation = torch.nn.Softmax(dim=-1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d * num_layers * hidden_size, num_classes),
            activation
        )

    def forward(self, input):
        output, (h_n, c_n) = self.backbone(input.payload)
        # output, h_n = self.backbone(input.payload)
        batch_size = h_n.shape[-2]
        h_n = h_n.view(batch_size, -1)
        return self.linear(h_n)


def preprocess_data():
    df = pd.read_csv(base_path / 'churn' / 'train.csv')
    enc = LabelEncoder()
    df['MCC'] = enc.fit_transform(df['MCC'])
    df['TRDATETIME'] = pd.to_datetime(df['TRDATETIME'], format='%d%b%y:%H:%M:%S')
    df = df[df['currency'] == 810]

    df_target = df[['cl_id', 'target_flag']].drop_duplicates()
    df_target.reset_index(inplace=True, drop=True)

    df_target_train, df_target_test = train_test_split(
        df_target, test_size=1000, stratify=df_target['target_flag'], random_state=142)
    df_target_train, df_target_valid = train_test_split(
        df_target_train, test_size=500, stratify=df_target_train['target_flag'], random_state=142)
    print('Split {} records to train: {}, valid: {}, test: {}'.format(
        *[len(df_) for df_ in [df_target, df_target_train, df_target_valid, df_target_test]]))

    df_trx = df[['cl_id', 'MCC', 'TRDATETIME', 'amount']]
    df_trx_train = pd.merge(df_trx, df_target_train['cl_id'], on='cl_id', how='inner')
    df_trx_valid = pd.merge(df_trx, df_target_valid['cl_id'], on='cl_id', how='inner')
    df_trx_test = pd.merge(df_trx, df_target_test['cl_id'], on='cl_id', how='inner')
    print('Split {} transactions to train: {}, valid: {}, test: {}'.format(
            *[len(df_) for df_ in [df_trx, df_trx_train, df_trx_valid, df_trx_test]]))
    
    preprocessor = PandasDataPreprocessor(
        col_id='cl_id',
        col_event_time='TRDATETIME',
        event_time_transformation='dt_to_timestamp',
        cols_category=['MCC'],
        cols_numerical=['amount'],
        return_records=False,
    )

    df_data_train = preprocessor.fit_transform(df_trx_train)
    df_data_valid = preprocessor.transform(df_trx_valid)
    df_data_test = preprocessor.transform(df_trx_test)

    df_data_train = pd.merge(df_data_train, df_target, on='cl_id')
    df_data_valid = pd.merge(df_data_valid, df_target, on='cl_id')
    df_data_test = pd.merge(df_data_test, df_target, on='cl_id')

    df_data_train = df_data_train.to_dict(orient='records')
    df_data_valid = df_data_valid.to_dict(orient='records')
    df_data_test = df_data_test.to_dict(orient='records')

    dataset_train = MemoryMapDataset(df_data_train)
    dataset_valid = MemoryMapDataset(df_data_valid)
    dataset_test = MemoryMapDataset(df_data_test)

    sup_data = PtlsDataModule(
        train_data=SeqToTargetDataset(dataset_train, target_col_name='target_flag', target_dtype=torch.long),
        valid_data=SeqToTargetDataset(dataset_valid, target_col_name='target_flag', target_dtype=torch.long),
        test_data=SeqToTargetDataset(dataset_test, target_col_name='target_flag', target_dtype=torch.long),
        train_batch_size=128,
        valid_batch_size=1024,
        train_num_workers=8,
    )

    return dataset_train, dataset_valid, dataset_test

def train_and_eval():

    with wandb.init():

        cfg = wandb.config

        dataset_train, dataset_valid, dataset_test = preprocess_data()

        sup_data = PtlsDataModule(
            train_data=SeqToTargetDataset(dataset_train, target_col_name='target_flag', target_dtype=torch.long),
            valid_data=SeqToTargetDataset(dataset_valid, target_col_name='target_flag', target_dtype=torch.long),
            test_data=SeqToTargetDataset(dataset_test, target_col_name='target_flag', target_dtype=torch.long),
            train_batch_size=128,
            valid_batch_size=1024,
            train_num_workers=8,
        )

        seq_encoder = RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings={
                    'MCC': {'in': 344, 'out': cfg['encoder']['mcc_out']},
                },
                numeric_values={
                    'amount': 'log',
                },
                embeddings_noise=0.001,
            ),
            hidden_size=cfg['encoder']['hidden_size'],
            is_reduce_sequence=False
        )

        model = RNNclassifier(input_size=cfg['encoder']['hidden_size'], 
                              hidden_size=cfg['rnn']['hidden_size'],
                              num_classes=2, num_layers=1, bidirectional=bool(cfg['rnn']['bidirectional']))

        sup_module = SequenceToTarget(
            seq_encoder=seq_encoder,
            head=model,
            loss=torch.nn.NLLLoss(),
            metric_list=[torchmetrics.Accuracy(), torchmetrics.F1Score()],
            optimizer_partial=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-3),
            lr_scheduler_partial=partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=5),
        )

        logger = pl_loggers.WandbLogger(name=cfg['experiment']['dataset'] + '_' + cfg['experiment']['model'] )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="max")

        trainer = pl.Trainer(
            max_epochs=cfg['experiment']['n_epochs'],
            gpus=1 if torch.cuda.is_available() else 0,
            logger=logger,
            callbacks=[early_stop_callback],
            check_val_every_n_epoch=10)
        
        trainer.fit(sup_module, sup_data)

        inference_dl = torch.utils.data.DataLoader(
            dataset=dataset_test,
            collate_fn=collate_feature_dict,
            shuffle=False,
            batch_size=1000,
            num_workers=4,
        )

        inf_module = InferenceModule(
            sup_module,
            model_out_name='log_prob',
        )

        df_predict = trainer.predict(inf_module, inference_dl)
        df_predict = pd.concat(df_predict, axis=0)

        df_predict[['log_prob_0000', 'log_prob_0001']] = df_predict[['log_prob_0000', 'log_prob_0001']].apply(np.exp, axis=0)

        y_pred = df_predict[[f'log_prob_{i:04d}' for i in range(2)]].values.argmax(axis=1)
        y_true = df_predict['target_flag'].values

        fscore = f1_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, df_predict['log_prob_0001'].values)

        logger.experiment.log({'Test f1-score': fscore, 'Test AUROC': auroc})


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default='abazarova')
    parser.add_argument('--project', type=str, default='coles')
    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--count', type=int, default=5)

    args = parser.parse_args()

    wandb.agent(args.sweep_id, function=train_and_eval,
                entity=args.entity, project=args.project, count=args.count)