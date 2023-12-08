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
    """Calculates classification metrics on the test dataset.

    Args:
    ----
        df_predict: pd.DataFrame object with columns
            ["target", "prob_0000", .., "prob_{num_classes - 1}"]
        task: str - either "binary" or "multiclass"
        num_classes: int - number of classes in the classification task

    Returns:
    -------
        (float, float, float, float, wandb.Image):
            ROC-AUC, Precision-Recall AUC, accuracy (with the threshold that maximizes TPR-FPR),
            F1-score, confusion matrix

    """
    y_true = df_predict["target"].values

    if task == "binary":
        probas = df_predict["prob_0001"].values

        auroc = roc_auc_score(y_true, probas)
        prauc = average_precision_score(y_true, df_predict["prob_0001"].values)

        fpr, tpr, thresholds = roc_curve(
            y_true, df_predict["prob_0001"].values, pos_label=1
        )

        optimal_th = thresholds[np.argmax(tpr - fpr)]
        y_pred = df_predict["prob_0001"].values > optimal_th

        fscore = f1_score(y_true, y_pred)

    if task == "multiclass":
        probas = df_predict[[f"prob_{i:04d}" for i in range(num_classes)]].values

        auroc = roc_auc_score(
            y_true,
            probas,
            multi_class="ovr",
        )

        average_precision = AveragePrecision(
            task=task, num_classes=num_classes, average="macro"
        )
        prauc = average_precision(
            torch.from_numpy(probas),
            torch.from_numpy(y_true),
        )

        y_pred = probas.argmax(axis=1)

        fscore = f1_score(y_true, y_pred, average="macro")

    acc = accuracy_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=np.arange(num_classes)
    )
    plt.switch_backend("agg")
    cmd = disp.plot()
    image = wandb.Image(cmd.figure_)
    plt.close()

    return auroc, prauc, acc, fscore, image
