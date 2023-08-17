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

from torchmetrics import Accuracy, F1Score, Precision, Recall, AveragePrecision
from ptls.nn import TrxEncoder, RnnSeqEncoder

from functools import partial
from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
from ptls.frames import PtlsDataModule

from ptls.data_load.utils import collate_feature_dict
import random
from ptls.frames.inference_module import InferenceModule
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    average_precision_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

base_path = Path("/home/COLES/poison/data")


class BestClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int = None,
        rnn_units: int = 128,  # C_out
        classifier_units: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = True,
        num_classes: int = 4,
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout2d(0.5)
        self.gru = self.backbone = torch.nn.GRU(
            input_size=input_size,
            hidden_size=rnn_units,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        self.linear1 = torch.nn.Linear(
            in_features=3 * d * rnn_units, out_features=classifier_units
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            in_features=classifier_units, out_features=num_classes
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * d * rnn_units, classifier_units),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_units, num_classes),
        )

    def forward(self, input):
        embs = input.payload  # (N, L, C)
        dropout_embs = self.dropout(embs)
        states, h_n = self.backbone(dropout_embs)  # (N, L, 2 * C_out), (2, N, C_out)

        rnn_max_pool = states.max(dim=1)[0]  # (N, 2 * C_out)
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]  # (N, 2 * C_out)
        batch_size = h_n.shape[-2]
        h_n = h_n.view(batch_size, -1)  # (N, 2 * C_out)

        output = torch.cat([rnn_max_pool, rnn_avg_pool, h_n], dim=-1)  # (N, 6 * C_out)
        drop = torch.nn.functional.dropout(output, p=0.5)
        logit = self.mlp(drop)
        probas = torch.nn.functional.softmax(logit, dim=1)

        return probas
    

class CNNclassifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        num_classes: int = 2,
        sequence_length: int = 300,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_layers = int(np.log2(in_channels)) - 1
        self.convolutions = torch.nn.Sequential(
            *[
                torch.nn.Conv1d(
                    2 ** (k + 1),
                    2**k,
                    kernel_size,
                    stride,
                    padding="same",
                    dilation=dilation,
                    bias=bias,
                )
                for k in range(self.num_layers, -1, -1)
            ]
        )

        self.activation = torch.nn.Softmax(dim=-1)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(sequence_length, num_classes),
            self.activation,
        )

    def forward(self, input):
        inp = torch.transpose(input.payload, 1, 2)  # (N, L, C) -> (N, C, L)
        l = inp.shape[-1]
        inp = torch.nn.functional.pad(inp, pad=(0, self.sequence_length - l))

        output = self.convolutions(inp)
        output = output.view(output.shape[0], -1)

        return self.linear(output)


def preprocess_data():
    df_target = pd.read_csv(base_path / 'age' / 'train_target.csv')

    df_target_train, df_target_test = train_test_split(
        df_target, test_size=7000, stratify=df_target['bins'], random_state=142)
    df_target_train, df_target_valid = train_test_split(
        df_target_train, test_size=3000, stratify=df_target_train['bins'], random_state=142)
    print('Split {} records to train: {}, valid: {}, test: {}'.format(
        *[len(df) for df in [df_target, df_target_train, df_target_valid, df_target_test]]))

    df_trx = pd.read_csv(base_path / 'age' / 'transactions_train.csv')

    df_trx_train = pd.merge(df_trx, df_target_train['client_id'], on='client_id', how='inner')
    df_trx_valid = pd.merge(df_trx, df_target_valid['client_id'], on='client_id', how='inner')
    df_trx_test = pd.merge(df_trx, df_target_test['client_id'], on='client_id', how='inner')
    print('Split {} transactions to train: {}, valid: {}, test: {}'.format(
        *[len(df) for df in [df_trx, df_trx_train, df_trx_valid, df_trx_test]]))

    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='trans_date',
        event_time_transformation='none',
        cols_category=['small_group'],
        cols_numerical=['amount_rur'],
        return_records=False,
    )

    df_data_train = preprocessor.fit_transform(df_trx_train)
    df_data_valid = preprocessor.transform(df_trx_valid)
    df_data_test = preprocessor.transform(df_trx_test)

    df_target = df_target.rename(columns={'bins': 'target_bin'})

    df_data_train = pd.merge(df_data_train, df_target, on='client_id')
    df_data_valid = pd.merge(df_data_valid, df_target, on='client_id')
    df_data_test = pd.merge(df_data_test, df_target, on='client_id')

    df_data_train = df_data_train.to_dict(orient='records')
    df_data_valid = df_data_valid.to_dict(orient='records')
    df_data_test = df_data_test.to_dict(orient='records')

    dataset_train = MemoryMapDataset(df_data_train)
    dataset_valid = MemoryMapDataset(df_data_valid)
    dataset_test = MemoryMapDataset(df_data_test)

    return dataset_train, dataset_valid, dataset_test


def train_and_eval():
    with wandb.init() as run:
        cfg = wandb.config

        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

        dataset_train, dataset_valid, dataset_test = preprocess_data()

        sup_data = PtlsDataModule(
            train_data=SeqToTargetDataset(dataset_train, target_col_name='target_bin', target_dtype=torch.long),
            valid_data=SeqToTargetDataset(dataset_valid, target_col_name='target_bin', target_dtype=torch.long),
            test_data=SeqToTargetDataset(dataset_test, target_col_name='target_bin', target_dtype=torch.long),
            train_batch_size=64,
            valid_batch_size=64,
            train_num_workers=8,
        )

        seq_encoder = RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings={
                    'small_group': {'in': 150, 'out': 32},
                },
                numeric_values={
                    'amount_rur': 'log',
                },
                embeddings_noise=0.001,
            ),
            hidden_size=cfg["encoder"]["hidden_size"],
            is_reduce_sequence=False
        )

        # model = BestClassifier(
        #     input_size=cfg["encoder"]["hidden_size"],
        #     rnn_units=cfg["encoder"]["rnn_units"],
        #     classifier_units=cfg["encoder"]["rnn_units"] // 2,
        # )

        model = CNNclassifier(
            in_channels=cfg["encoder"]["hidden_size"],
            kernel_size=cfg["cnn"]["kernel_size"],
            num_classes=4,
        )
        
        sup_module = SequenceToTarget(
            seq_encoder=seq_encoder,
            head=model,
            loss=torch.nn.CrossEntropyLoss(),
            metric_list=[
                Accuracy(task="multiclass", num_classes=4),
                F1Score(task="multiclass", num_classes=4),
                Precision(task="multiclass", num_classes=4),
                Recall(task="multiclass", num_classes=4),
                AveragePrecision(task="multiclass", num_classes=4),
            ],
            optimizer_partial=partial(torch.optim.AdamW, lr=3e-4),
            lr_scheduler_partial=partial(
                torch.optim.lr_scheduler.StepLR, step_size=40, gamma=0.5
            ),
        )

        logger = pl_loggers.WandbLogger(
            name=cfg["experiment"]["dataset"] + "_" + cfg["experiment"]["model"]
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=cfg["experiment"]["n_epochs"],
            gpus=1 if torch.cuda.is_available() else 0,
            logger=logger,
            callbacks=[early_stop_callback],
            check_val_every_n_epoch=10,
        )

        trainer.fit(sup_module, sup_data)
        torch.cuda.empty_cache()

        ds = cfg["experiment"]["dataset"]
        mdl = cfg["experiment"]["model"]
        torch.save(
            sup_module.state_dict(), f"../notebooks/saves/{ds}_{mdl}_{run.name}.pth"
        )

        inference_dl = torch.utils.data.DataLoader(
            dataset=dataset_test,
            collate_fn=collate_feature_dict,
            shuffle=False,
            batch_size=32,
            num_workers=4,
        )

        inf_module = InferenceModule(
            sup_module,
            model_out_name="prob",
        )

        df_predict = trainer.predict(inf_module, inference_dl)
        df_predict = pd.concat(df_predict, axis=0)

        y_true = df_predict["target_bin"].values

        auroc = roc_auc_score(y_true, df_predict[[f'prob_{i:04d}' for i in range(4)]].values, multi_class='ovr')

        average_precision = AveragePrecision(task="multiclass", num_classes=4, average='macro')
        prauc = average_precision(torch.from_numpy(df_predict[[f'prob_{i:04d}' for i in range(4)]].values),
                                                    torch.from_numpy(y_true))

        y_pred = df_predict[[f'prob_{i:04d}' for i in range(4)]].values.argmax(axis=1)

        fscore = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
        plt.switch_backend("agg")
        cmd = disp.plot()

        logger.experiment.log(
            {
                "Test f1-score": fscore,
                "Test AUROC": auroc,
                "Test accuracy": acc,
                "Test PR-AUC": prauc,
            }
        )
        logger.experiment.log({"Confusion matrix, test": wandb.Image(cmd.figure_)})
        plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="abazarova")
    parser.add_argument("--project", type=str, default="coles")
    parser.add_argument("--sweep_id", type=str)
    parser.add_argument("--count", type=int, default=5)

    args = parser.parse_args()

    wandb.agent(
        args.sweep_id,
        function=train_and_eval,
        entity=args.entity,
        project=args.project,
        count=args.count,
    )
