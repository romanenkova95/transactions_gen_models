import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict

from ptls.data_load.padded_batch import PaddedBatch
from tqdm import tqdm
from ptls.data_load.utils import collate_feature_dict

class PoolingModel(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions 
        representations with pooling of information of different users.
    """

    def __init__(
        self,
        train_data: List[Dict],
        backbone: nn.Module,
        agregating_model: nn.Module = None,
        pooling_type: str = "mean",
        backbone_embd_size: int = None,
        backbone_output_type: str = "tensor",
        max_users_in_train_dataloader=3000,
        min_seq_length = 15,
        columns = ['event_time', 'mcc_code', 'amount']
    ) -> None:
        """Initialize method for PoolingModel

        Args:
            backbone ():  Local embeding model
            argegating_model (torch.nn.Module): Model for agregating pooled embeddings
            train_data (train_dataloader from CustomColesValidationDataset): DataLoader for calculating global
                embedings from local sequences, train_batch_size need to be set equal to 1
            pooling_type (str): "max" or "mean", type of pooling
        """
        super().__init__()

        assert backbone_output_type in [
            "tensor",
            "padded_batch",
        ], "Unknown output type of the backbone model"

        self.backbone = backbone
        self.backbone_embd_size = backbone_embd_size

        # freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.agregating_model = agregating_model
        if agregating_model is None:
            self.agregating_model_emb_dim = 0

        self.backbone_output_type = backbone_output_type
        self.min_seq_length = min_seq_length
        self.columns = columns
        self.pooling_type = pooling_type

        self.pooled_embegings_dataset = self.make_pooled_embegings_dataset(
            train_data, max_users_in_train_dataloader
        )

    def prepare_data_for_one_user(self, x):
        """
        Args:
            x(Dict): data from one user
        """
        resulting_user_data = []
        for i in range(self.min_seq_length, len(x[self.columns[0]])):
            data_for_timestamp = {}
            for column in self.columns:
                data_for_timestamp[column] = x[column][:i]
            resulting_user_data.append(data_for_timestamp)
        resulting_user_data = collate_feature_dict(resulting_user_data)
        return resulting_user_data

    def make_pooled_embegings_dataset(
        self,
        train_data: List[Dict],
        max_users_in_train_dataloader: int,
    ) -> dict[int, torch.Tensor]:
        """Creation of pooled embeding dataset. This function for each timestamp get
        sequences in dataset which ends close to this timestamp,
        make local embedding out of them and pool them together

        Args:
            train_dataloader (List[Dict]): data for calculating global embedings from local sequences

        Return:
            pooled_embegings_dataset(dict): dictionary containing timestamps and pooling vectors for that timestamps
        """
        data = {}
        all_times = set()
        for i, x, in tqdm(enumerate(train_data)):
            data[i] = {}
            if len(x[self.columns[0]]) <= self.min_seq_length:
                continue
            prepared_x = self.prepare_data_for_one_user(x)
            embs = self.backbone(prepared_x).detach().cpu().numpy()
            times = prepared_x.payload["event_time"].max(1).values.cpu().numpy()
            all_times.update(times)
            for emb, time in zip(embs, times):
                data[i][time] = emb
            if i >= max_users_in_train_dataloader:
                break

        all_times = list(all_times)
        all_times.sort()
        pooled_embegings_dataset = {}
        for time in all_times:
            embs_for_this_time = []
            for i in data.keys():
                user_times = np.sort(list(data[i].keys()))

                indexes = np.where(np.array(user_times) < time)[0]
                if len(indexes) == 0:
                        continue
                closest_time = user_times[indexes[-1]]
                closest_emb = data[i][closest_time]
                embs_for_this_time.append(closest_emb)
            if len(embs_for_this_time) > 0:
                pooled_embegings_dataset[time] = np.stack(embs_for_this_time)

        self.times = np.sort(list(pooled_embegings_dataset.keys()))

        return pooled_embegings_dataset

    def forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Forward method."""

        batch_of_local_embedings = self.backbone(batch)

        batch_of_global_poolings = torch.tensor([]).to(self.device)
        for event_time_seq, user_emb in zip(batch.payload["event_time"], batch_of_local_embedings):
            local_pooled_emb = self.make_local_pooled_embedding(
                event_time_seq.max().item(), user_emb
            )
            local_pooled_emb = local_pooled_emb.to(self.device)
            batch_of_global_poolings = torch.cat(
                (batch_of_global_poolings, local_pooled_emb.unsqueeze(0))
            )

        out = torch.cat(
            (
                batch_of_local_embedings,
                batch_of_global_poolings.to(batch_of_local_embedings.device),
            ),
            dim=1,
        )

        return out

    def make_local_pooled_embedding(self, time: int, user_emb: torch.Tensor) -> torch.Tensor:
        """Function that find the most close timestamp in self.pooled_embegings_dataset
        and return the pooling vector at this timestamp.

        Args:
            time (int): timepoint for which we are looking for pooling vector

        Return:
            pooled_vector (torch.Tensor): pooling vector for given timepoint

        """
        
        indexes = np.where(np.array(self.times) < time)[0]

        if len(indexes) == 0:
                pooled_vector = torch.rand(self.backbone_embd_size)
        else:
            closest_time = self.times[indexes[-1]]
            vectors = self.pooled_embegings_dataset[closest_time]
            if self.pooling_type == "mean":
                pooled_vector = torch.tensor(np.mean(vectors, axis=0))
            elif self.pooling_type == "max":
                pooled_vector = torch.tensor(np.max(vectors, axis=0))
            elif self.pooling_type == "attention":
                vectors = torch.tensor(vectors).to(user_emb.device)
                dot_prod = vectors @ user_emb.unsqueeze(0).transpose(1, 0)
                sortmax_dot_prod = nn.functional.softmax(dot_prod, 0)
                pooled_vector =  (sortmax_dot_prod * vectors).sum(axis=0)
            else:
                raise ValueError("Unsupported pooling type.")
        return pooled_vector

    def get_emb_dim(self):
        return 2 * self.backbone_embd_size
