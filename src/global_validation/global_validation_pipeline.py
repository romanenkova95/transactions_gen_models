"""Global target validation script."""
import warnings
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.data_load.utils import collate_feature_dict
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
)

from src.preprocessing import preprocess
from src.utils.logging_utils import get_logger


def global_target_validation(
    data: tuple[list[dict], list[dict], list[dict]],
    cfg_encoder: DictConfig,
    cfg_validation: DictConfig,
    encoder_name: str,
) -> dict:
    """Full pipeline for the sequence encoder validation.

    Args:
    ----
        data (tuple[list[dict], list[dict], list[dict]]):
            train, val & test sets
        cfg_encoder (DictConfig):
            encoder config (specified in 'config/encoder')
        cfg_validation (DictConfig):
            Validation config (specified in 'config/validation')
        encoder_name (str):
            The name of encoder to use when loading weights.

    Returns:
    -------
        results (dict):      dict with test metrics
    """
    logger = get_logger(name=__name__)

    train, val, test = data
    logger.info("Instantiating the sequence encoder")

    # load pretrained sequence encoder
    sequence_encoder = instantiate(cfg_encoder, is_reduce_sequence=True)
    encoder_state_dict_path = Path(f"saved_models/{encoder_name}.pth")
    if encoder_state_dict_path.exists():
        sequence_encoder.load_state_dict(torch.load(encoder_state_dict_path))
    else:
        warnings.warn("No encoder state dict found! Validating with random weights...")

    logger.info("Processing train sequences")

    # get representations of sequences from train + val part
    embeddings, targets = embed_data(
        sequence_encoder, train + val, **cfg_validation["embed_data"]
    )
    N = len(embeddings)
    indices = np.arange(N)

    logger.info("Processing test sequences")

    # get representations of sequences from test part
    embeddings_test, targets_test = embed_data(
        sequence_encoder, test, **cfg_validation["embed_data"]
    )

    # bootstrap sample
    bootstrap_inds = np.random.choice(indices, size=N, replace=True)
    embeddings_train, targets_train = (
        embeddings[bootstrap_inds],
        targets[bootstrap_inds],
    )

    # evaluate trained model
    metrics = eval_embeddings(
        embeddings_train,
        targets_train,
        embeddings_test,
        targets_test,
        cfg_validation["model"],
    )

    return metrics


def embed_data(
    seq_encoder: torch.nn.Module,
    dataset: list[dict],
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed sequences, and return them along with the corresponding targets.

    Args:
    ----
        seq_encoder (nn.Module): Pretrained sequence encoder
        dataset (list):          Dataset
        batch_size (int):        How many sequences are processed in a batch
        device (str):            Device

    Returns:
    -------
        features (np.array): Array of shape (n, embedding_dim) with embeddings of sequences
        targets (np.array):  Array of shape (n,) with targets of sequences
    """
    seq_encoder.to(device)
    seq_encoder.eval()

    with torch.no_grad():
        embeddings_all, targets_all = [], []
        num_iter = int(np.ceil(len(dataset) / batch_size))

        for i in range(num_iter):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            batch_collated = collate_feature_dict(batch).to(device)

            embeddings = seq_encoder(batch_collated).detach().cpu()
            embeddings_all += [*embeddings]

            targets_all += [batch_collated.payload["global_target"].detach().cpu()]

    features = torch.vstack(embeddings_all).numpy()
    targets = torch.cat(targets_all).numpy()

    return features, targets


def eval_embeddings(
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    test_embeds: np.ndarray,
    test_labels: np.ndarray,
    model_config: DictConfig,
) -> dict[str, float]:
    """Trains and evaluates a simple classifier on the embedded data.

    Args:
    ----
        train_embeds (np.ndarray): Embeddings of sequences from train set
        train_labels (np.ndarray): Labels of sequences from train set
        test_embeds (np.ndarray):  Embeddings of sequences from test set
        test_labels (np.ndarray):  Labels of sequences from test set
        model_config (DictConfig):   Config of the model to be fitted

    Returns:
    -------
        results (dict): Dictionary with test metrics values
    """
    nunique = len(np.unique(train_labels))
    if nunique > 2:
        objective = "multiclass"
        lgbm_metric = "multi_error"
        metrics = {
            "AUROC": AUROC(task="multiclass", num_classes=nunique),
            "PR-AUC": AveragePrecision(task="multiclass", num_classes=nunique),
            "Accuracy": Accuracy(task="multiclass", num_classes=nunique),
            "F1Score": F1Score(task="multiclass", num_classes=nunique),
        }
    else:
        objective = "binary"
        lgbm_metric = "accuracy"
        metrics = {
            "AUROC": AUROC(task="binary"),
            "PR-AUC": AveragePrecision(task="binary"),
            "Accuracy": Accuracy(task="binary"),
            "F1Score": F1Score(task="binary"),
        }

    model = instantiate(model_config, metric=lgbm_metric, objective=objective)

    model.fit(train_embeds, train_labels)
    y_pred_test = model.predict_proba(test_embeds)

    if objective == "binary":
        y_pred_test = y_pred_test[:, 1]

    results = {}
    for name, metric in metrics.items():
        results[name] = metric(
            torch.Tensor(y_pred_test), torch.LongTensor(test_labels)
        ).item()

    return results
