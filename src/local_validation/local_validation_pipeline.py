"""Local targets validation script. """

from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames import PtlsDataModule

from src.utils.logging_utils import get_logger
from src.utils.data_utils.prepare_dataset import prepare_dataset
from src.local_validation import LocalValidationModel

def local_target_validation(cfg_preprop: DictConfig, cfg_validation: DictConfig) -> pd.DataFrame:
    """Full pipeline for the sequence encoder local validation. 

    Args:
        cfg_preprop (DictConfig):    Dataset config (specified in the 'config/dataset')
        cfg_validation (DictConfig): Validation config (specified in the 'config/validation')
    
    Returns:
        results (pd.DataFrame):      Dataframe with test metrics for each run
    """
    logger = get_logger(name=__name__)

<<<<<<< HEAD
    def __init__(
        self,
        backbone: nn.Module,
        backbone_embd_size: int,
        hidden_size: int,
        learning_rate: float = 1e-3,
        backbone_output_type: str = "tensor",
        use_lstm: bool = False,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.
=======
    dataset = prepare_dataset(cfg_preprop, logger)
>>>>>>> origin/main

    # train val test split
    valid_size = cfg_preprop["coles"]["valid_size"]
    test_size = cfg_preprop["coles"]["test_size"]

    train, val_test = train_test_split(
        dataset,
        test_size=valid_size+test_size,
        random_state=cfg_preprop["coles"]["random_state"]
    )

    val, test = train_test_split(
        val_test,
        test_size=test_size/(valid_size+test_size),
        random_state=cfg_preprop["coles"]["random_state"]
    )

    logger.info("Instantiating the sequence encoder")
    # load pretrained sequence encoder
    sequence_encoder = instantiate(cfg_validation["sequence_encoder"])
    sequence_encoder.load_state_dict(torch.load(cfg_validation["path_to_state_dict"]))

<<<<<<< HEAD
        self.use_lstm = use_lstm

        if use_lstm:
            self.lstm = torch.nn.LSTM(
                input_size=backbone_embd_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                bidirectional=lstm_bidirectional,
            )

            self.pred_head = nn.Sequential(
                nn.Linear(
                    lstm_hidden_size * (lstm_bidirectional + 1) + backbone_embd_size,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        else:
            self.pred_head = nn.Sequential(
                nn.Linear(backbone_embd_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
=======
    data_train = MemoryMapDataset(train, [SeqLenFilter(cfg_validation["model"]["seq_len"])])
    data_val = MemoryMapDataset(val, [SeqLenFilter(cfg_validation["model"]["seq_len"])])
    data_test = MemoryMapDataset(test, [SeqLenFilter(cfg_validation["model"]["seq_len"])])

    train_dataset: ColesDataset = instantiate(cfg_validation["dataset"], data=data_train)
    val_dataset: ColesDataset = instantiate(cfg_validation["dataset"], data=data_val)
    test_dataset: ColesDataset = instantiate(cfg_validation["dataset"], data=data_test)

    datamodule: PtlsDataModule = instantiate(
        cfg_validation["datamodule"],
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
    )

    results = []
    for i in range(cfg_validation["n_runs"]):
        logger.info(f'Training LocalValidationModel. Run {i+1}/{cfg_validation["n_runs"]}')

        seed_everything(i)

        valid_model: LocalValidationModel = instantiate(
            cfg_validation["model"],
            backbone=sequence_encoder 
        )
>>>>>>> origin/main

        val_trainer: Trainer = instantiate(cfg_validation["trainer"])

        val_trainer.fit(valid_model, datamodule)
        torch.save(valid_model.state_dict(), f'saved_models/validation_head_{i}.pth')

        metrics = val_trainer.test(valid_model, datamodule)
        results.append(metrics)

<<<<<<< HEAD
        self.backbone_output_type = backbone_output_type

    def forward(self, inputs: PaddedBatch) -> torch.Tensor:
        """Do forward pass through the global validation model.

        Args:
            inputs (PaddedBatch) - inputs if ptls format

        Returns:
            torch.Tensor of predicted local targets
        """
        out = self.backbone(inputs)
        if self.backbone_output_type == "padded_batch":
            out = out.payload
        if self.use_lstm:
            lstm_out = self.lstm(out)[0]
            out = torch.cat((out, lstm_out), dim=1)
        out = self.pred_head(out).squeeze(-1)
        return out

    def training_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Training step of the LocalValidationModel."""
        inputs, labels = batch
        preds = self.forward(inputs)

        train_loss = self.loss(preds, labels.float())
        train_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return {"loss": train_loss, "acc": train_accuracy}

    def validation_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Validation step of the LocalValidationModel."""
        inputs, labels = batch

        preds = self.forward(inputs)

        val_loss = self.loss(preds, labels.float())
        val_accuracy = ((preds.squeeze() > 0.5).long() == labels).float().mean()

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return {"loss": val_loss, "acc": val_accuracy}

    def test_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """Test step of the LocalValidationModel."""
        inputs, labels = batch
        preds = self.forward(inputs)

        dict_out = {"preds": preds, "labels": labels}
        for name, metric in self.metrics.items():
            metric.to(inputs.device)
            metric.update(preds, labels)

            dict_out[name] = metric.compute().item()

        return dict_out

    def on_test_epoch_end(self) -> dict[str, float]:
        """Collect test_step outputs and compute test metrics for the whole test dataset."""
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
            self.log(name, metric.compute())
        return results

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer for the LocalValidationModel."""
        opt = torch.optim.Adam(self.pred_head.parameters(), lr=self.lr)
        return opt
=======
    return pd.DataFrame(results)
>>>>>>> origin/main
