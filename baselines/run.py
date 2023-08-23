import argparse
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from ptls.data_load.utils import collate_feature_dict
from ptls.frames import PtlsDataModule
from ptls.frames.inference_module import InferenceModule
from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
from ptls.nn import RnnSeqEncoder, TrxEncoder
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, AveragePrecision, F1Score, Precision, Recall

from metrics import val_metrics
from models import BestClassifier, CNNClassifier, RNNClassifier
from preprocessing import load_dataset


def train_and_eval():
    with wandb.init() as run:
        cfg = wandb.config

        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

        dataset_name = cfg["experiment"]["dataset"]
        model_name = cfg["experiment"]["model"]

        num_classes = 4 if dataset_name == "age" else 2
        task = "multiclass" if dataset_name == "age" else "binary"

        dataset_train, dataset_valid, dataset_test = load_dataset(dataset_name)

        sup_data = PtlsDataModule(
            train_data=SeqToTargetDataset(
                dataset_train, target_col_name="target", target_dtype=torch.long
            ),
            valid_data=SeqToTargetDataset(
                dataset_valid, target_col_name="target", target_dtype=torch.long
            ),
            test_data=SeqToTargetDataset(
                dataset_test, target_col_name="target", target_dtype=torch.long
            ),
            train_batch_size=128,
            valid_batch_size=1024,
            train_num_workers=8,
        )

        seq_encoder = RnnSeqEncoder(
            trx_encoder=TrxEncoder(
                embeddings={
                    "category": {
                        "in": cfg["encoder"]["dict_size"],
                        "out": cfg["encoder"]["mcc_out"],
                    },
                },
                numeric_values={
                    "amount": "log",
                },
                embeddings_noise=0.001,
            ),
            hidden_size=cfg["encoder"]["hidden_size"],
            is_reduce_sequence=False,
        )

        model = BestClassifier(
            input_size=cfg["encoder"]["hidden_size"],
            rnn_units=cfg["encoder"]["rnn_units"],
            classifier_units=cfg["encoder"]["rnn_units"] // 2,
            num_classes=num_classes,
        )

        weight = torch.tensor([1.0, 25.0]) if dataset_name == "default" else None

        sup_module = SequenceToTarget(
            seq_encoder=seq_encoder,
            head=model,
            loss=torch.nn.CrossEntropyLoss(weight=weight),
            metric_list=[
                Accuracy(task=task, num_classes=num_classes, average="macro"),
                F1Score(task=task, num_classes=num_classes, average="macro"),
                Precision(task=task, num_classes=num_classes, average="macro"),
                Recall(task=task, num_classes=num_classes, average="macro"),
                AveragePrecision(task=task, num_classes=num_classes, average="macro"),
            ],
            optimizer_partial=partial(torch.optim.AdamW, lr=3e-4),
            lr_scheduler_partial=partial(
                torch.optim.lr_scheduler.StepLR, step_size=40, gamma=0.5
            ),
        )

        logger = pl_loggers.WandbLogger(name=f"{dataset_name}_{model_name}")
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

        torch.save(
            sup_module.state_dict(),
            f"saves/{dataset_name}_{model_name}_{run.name}.pth",
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

        auroc, prauc, acc, fscore, conf_matrix = val_metrics(
            df_predict, task=task, num_classes=num_classes
        )

        logger.experiment.log(
            {
                "Test f1-score": fscore,
                "Test AUROC": auroc,
                "Test accuracy": acc,
                "Test PR-AUC": prauc,
            }
        )
        logger.experiment.log({"Confusion matrix, test": conf_matrix})


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
