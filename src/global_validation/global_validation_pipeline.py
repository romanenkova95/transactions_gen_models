"""Global target validation script"""
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np

import torch

from ptls.data_load.utils import collate_feature_dict

from torchmetrics.classification import (
    F1Score,
    AUROC,
    AveragePrecision,
    Accuracy,
)

from src.utils.logging_utils import get_logger
from src.preprocessing import preprocess


def global_target_validation(cfg_preprop: DictConfig, cfg_validation: DictConfig) -> pd.DataFrame:
    """Full pipeline for the sequence encoder validation. 

    Args:
        cfg_preprop (DictConfig):    Preprocessing config (specified in the 'config/preprocessing')
        cfg_validation (DictConfig): Validation config (specified in the 'config/validation')
    
    Returns:
        results (pd.DataFrame):      Dataframe with test metrics for each run
    """
    logger = get_logger(name=__name__)
    
    train, val, test = preprocess(cfg_preprop)
    logger.info("Instantiating the sequence encoder")

    # load pretrained sequence encoder
    sequence_encoder = instantiate(cfg_validation["sequence_encoder"])
    sequence_encoder.load_state_dict(torch.load(cfg_validation["path_to_state_dict"]))

    logger.info("Processing train sequences")

    # get representations of sequences from train + val part
    embeddings, targets = embed_data(sequence_encoder, train + val, **cfg_validation["embed_data"])
    N = len(embeddings)
    indices = np.arange(N)

    logger.info("Processing test sequences")

    # get representations of sequences from test part
    embeddings_test, targets_test = embed_data(
        sequence_encoder,
        test,
        **cfg_validation["embed_data"]
    )

    results = []
    for i in range(cfg_validation["n_runs"]):
        logger.info(f'Training classifier. Run {i+1}/{cfg_validation["n_runs"]}')

        # bootstrap sample
        bootstrap_inds = np.random.choice(indices, size=N, replace=True)
        embeddings_train, targets_train = embeddings[bootstrap_inds], targets[bootstrap_inds]

        # evaluate trained model
        metrics = eval_embeddings(
            embeddings_train,
            targets_train,
            embeddings_test,
            targets_test,
            cfg_validation["model"]
        )

        results.append(metrics)

    return pd.DataFrame(results)


def embed_data(
        seq_encoder: torch.nn.Module,
        dataset:     list[dict],
        batch_size:  int = 64,
        device:      str = "cuda",
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns embeddings of sequences and corresponding targets.

    Args:
        seq_encoder (nn.Module): Pretrained sequence encoder
        dataset (list):          Dataset
        batch_size (int):        How many sequences are processed in a batch
        device (str):            Device

    Returns:
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
        train_embeds: torch.Tensor,
        train_labels: torch.Tensor,
        test_embeds:  torch.Tensor,
        test_labels:  torch.Tensor,
        model_config: DictConfig
    ) -> dict[str, float]:
    """
    Trains and evaluates a simple classifier on the embedded data.

    Args:
        train_embeds (torch.Tensor): Embeddings of sequences from train set 
        train_labels (torch.Tensor): Labels of sequences from train set 
        test_embeds (torch.Tensor):  Embeddings of sequences from test set 
        test_labels (torch.Tensor):  Labels of sequences from test set
        model_config (DictConfig):   Config of the model to be fitted

    Returns:
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
            torch.Tensor(y_pred_test),
            torch.LongTensor(test_labels)
        ).item()

    return results
