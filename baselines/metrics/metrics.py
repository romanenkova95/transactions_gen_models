import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torchmetrics import AveragePrecision


def val_metrics(df_predict, task="binary", num_classes=2):
    y_true = df_predict["target"].values

    if task == "binary":
        probas = df_predict["prob_0001"].values

        auroc = roc_auc_score(y_true, probas)
        prauc = average_precision_score(y_true, df_predict["prob_0001"].values)

        fpr, tpr, thresholds = roc_curve(
            y_true, df_predict["prob_0001"].values, pos_label=1
        )

        th = thresholds[np.argmax(tpr - fpr)]
        y_pred = df_predict["prob_0001"].values > th

        fscore = f1_score(y_true, y_pred)

    if task == "multiclass":
        probas = df_predict[[f"prob_{i:04d}" for i in range(num_classes)]].values

        auroc = roc_auc_score(
            y_true,
            probas,
            multi_class="ovr",
        )

        average_precision = AveragePrecision(task=task, num_classes=num_classes, average="macro")
        prauc = average_precision(
            torch.from_numpy(probas),
            torch.from_numpy(y_true),
        )

        y_pred = probas.argmax(axis=1)

        fscore = f1_score(y_true, y_pred, average="macro")

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.arange(num_classes)
    )
    plt.switch_backend("agg")
    cmd = disp.plot()
    im = wandb.Image(cmd.figure_)
    plt.close()

    return auroc, prauc, acc, fscore, im
