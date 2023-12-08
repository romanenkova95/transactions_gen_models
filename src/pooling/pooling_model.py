import warnings
import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from ptls.data_load.padded_batch import PaddedBatch
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.feature_dict import FeatureDict


class PoolingModel(nn.Module):
    """PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions
    representations with pooling of information of different users.
    """

    def __init__(
        self,
        train_data: list[dict],
        backbone: nn.Module,
        pooling_type: str = "mean",
        backbone_embd_size: int = None,
        max_users_in_train_dataloader: int = 3000,
        min_seq_length: int = 15,
        max_seq_length: int = 1000,
        max_embs_per_user: int = 1000,
        init_device: str = "cuda",
        freeze: bool = True,
    ) -> None:
        """Initialize method for PoolingModel.

        Args:
        ----
            train_data (list[dict]): Dataset for calculating embedings
            backbone (nn.Module):  Local embeding model
            pooling_type (str): "max", "mean", "attention" or "learnable_attention", type of pooling
            backbone_embd_size (int): Size of local embedings from backbone model
            max_users_in_train_dataloader (int): Maximum number of users to save
                in self.embegings_dataset
            min_seq_length (int): Minimum length of sequence for user
                in self.embegings_dataset preparation
            max_seq_length (int): Maximum length of sequence for user
                in self.embegings_dataset preparation
            max_embs_per_user (int): How many datapoints to take from one user
            in self.embegings_dataset preparation
            init_device (str): Name of device to use during initialization
            freeze (bool): Flag
        """
        super().__init__()

        if pooling_type not in ["mean", "max", "attention", "learnable_attention"]:
            raise ValueError("Unknown pooling type.")

        self.backbone = backbone.to(init_device)
        self.backbone.eval()
        self.backbone_embd_size = backbone_embd_size

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pooling_type = pooling_type
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.max_embs_per_user = max_embs_per_user

        self.embegings_dataset = self.make_pooled_embegings_dataset(
            train_data, max_users_in_train_dataloader, init_device
        )

        if pooling_type == "learnable_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.backbone_embd_size, self.backbone_embd_size
            )
        else:
            self.learnable_attention_matrix = None

    def prepare_data_for_one_user(self, x, device):
        """Function for preparing one user's embedings and last times for this embedings.

        Args:
        ----
            x (dict): Data from one user
            device (str): Name of device to use

        Return:
        ------
            embs (np.array): Embeddings of slices of one user
            times (np.array): Times of last transaction in slices of ine user
        """
        resulting_user_data = []
        indexes = np.arange(self.min_seq_length, len(x["event_time"]))
        if len(indexes) >= self.max_embs_per_user:
            indexes = np.random.choice(indexes, self.max_embs_per_user, replace=False)
            indexes = np.sort(indexes)
        times = x["event_time"][indexes - 1].cpu().numpy()
        for i in indexes:
            start_index = 0 if i < self.max_seq_length else i - self.max_seq_length
            data_for_timestamp = FeatureDict.seq_indexing(x, range(start_index, i))
            resulting_user_data.append(data_for_timestamp)
        resulting_user_data = collate_feature_dict(resulting_user_data)
        embs = self.backbone(resulting_user_data.to(device)).detach().cpu().numpy()
        return embs, times

    def make_pooled_embegings_dataset(
        self, train_data: list[dict], max_users_in_train_dataloader: int, device: str
    ) -> dict[int, torch.Tensor]:
        """Creation of pooled embeding dataset. This function for each timestamp get
        sequences in dataset which ends close to this timestamp,
        make local embedding out of them and pool them together

        Args:
        ----
            train_data (list[dict]): data for calculating global embedings
                from local sequences
            device (str): name of device to use

        Return:
        ------
            embegings_dataset(dict): dictionary containing timestamps and pooling
                vectors for that timestamps
        """
        data = []
        all_times = set()
        for x in tqdm(train_data):
            # check if the user's sequence is not long enough
            if len(x["event_time"]) <= self.min_seq_length:
                continue

            data.append({})
            embs, times = self.prepare_data_for_one_user(x, device)
            all_times.update(times)
            for emb, time in zip(embs, times):
                data[-1][time] = emb

            # check if the number of users is enough
            if len(data) >= max_users_in_train_dataloader:
                break

        all_times = list(all_times)
        all_times.sort()
        embegings_dataset = self.reshape_time_user_dataset(all_times, data)
        self.times = np.sort(list(embegings_dataset.keys()))

        return embegings_dataset

    def reshape_time_user_dataset(
        self, all_times: list, data: list[dict]
    ) -> dict[int, torch.Tensor]:
        """Reshaping of time-users-embeddings data from list of users with Dicts
        with time keys to dict with time keys and values with concatenated users
        embeddings

        Args:
        ----
            all_times (list): list of all time points that can be found in data
            data (list[dict]): list of users containing the time as a keys and
            embeddings as values

        Return:
        ------
            embegings_dataset(dict): dictionary containing timestamps and pooling
                vectors for that timestamps
        """
        embegings_dataset = {}
        for time in tqdm(all_times):
            embs_for_this_time = []
            for user_data in data:
                user_times = list(user_data.keys())
                user_times.sort()
                user_times = np.array(user_times)

                index = np.searchsorted(user_times, time, "right") - 1
                if index < 0:
                    continue

                closest_time = user_times[index]
                closest_emb = user_data[closest_time]
                embs_for_this_time.append(closest_emb)
            if len(embs_for_this_time) > 0:
                embegings_dataset[time] = np.stack(embs_for_this_time)

        return embegings_dataset

    def local_forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Local forward method (just pass batch through the backbone).

        Args:
        ----
            batch (PaddedBatch): batch of data

        Return:
        ------
            out (torch.Tensor): embedings of data in batch
        """
        out = self.backbone(batch)
        return out

    def global_forward(
        self, batch: PaddedBatch, batch_of_local_embedings: torch.Tensor
    ) -> torch.Tensor:
        """Global forward method ().

        Args:
        ----
            batch (PaddedBatch): batch of data
            batch_of_local_embedings (torch.Tensor): embedings of data in batch

        Return:
        ------
            batch_of_global_poolings (torch.Tensor): global embedings for
                last times for data in batch
        """
        batch_of_global_poolings = []
        for event_time_seq, user_emb in zip(
            batch.payload["event_time"], batch_of_local_embedings
        ):
            local_pooled_emb = self.make_local_pooled_embedding(
                event_time_seq.max().item(), user_emb
            )
            batch_of_global_poolings.append(local_pooled_emb)
        batch_of_global_poolings = torch.stack(batch_of_global_poolings)

        batch_of_global_poolings = batch_of_global_poolings

        return batch_of_global_poolings

    def forward(self, batch: PaddedBatch) -> torch.Tensor:
        """Forward method (makes local and global forward and concatenate them).

        Args:
        ----
            batch (PaddedBatch): batch of data

        Return:
        ------
            out (torch.Tensor): concatenated local and global forward outputs
        """
        batch_of_local_embedings = self.local_forward(batch)
        batch_of_global_poolings = self.global_forward(batch, batch_of_local_embedings)

        out = torch.cat(
            (
                batch_of_local_embedings,
                batch_of_global_poolings.to(batch_of_local_embedings.device),
            ),
            dim=1,
        )

        return out

    def make_local_pooled_embedding(
        self, time: int, user_emb: torch.Tensor
    ) -> torch.Tensor:
        """Function that find the most close timestamp in self.embegings_dataset
        and return the pooling vector at this timestamp.

        Args:
        ----
            time (int): timepoint for which we are looking for pooling vector

        Return:
        ------
            pooled_vector (torch.Tensor): pooling vector for given timepoint

        """
        index = np.searchsorted(self.times, time, "right") - 1
        if index < 0:
            warnings.warn(
                "Attention! Given data was before than any in train dataset. Pooling vector is set to random."
            )
            pooled_vector = torch.rand(self.backbone_embd_size)

        else:
            closest_time = self.times[index]
            vectors = self.embegings_dataset[closest_time]

            if self.pooling_type == "mean":
                pooled_vector = torch.Tensor(np.mean(vectors, axis=0))
            elif self.pooling_type == "max":
                pooled_vector = torch.Tensor(np.max(vectors, axis=0))
            elif self.pooling_type == "attention":
                vectors = torch.Tensor(vectors).to(user_emb.device)
                dot_prod = vectors @ user_emb.unsqueeze(0).transpose(1, 0)
                sortmax_dot_prod = nn.functional.softmax(dot_prod, 0)
                pooled_vector = (sortmax_dot_prod * vectors).sum(axis=0)
            elif self.pooling_type == "learnable_attention":
                vectors = torch.Tensor(vectors).to(user_emb.device)
                vectors = self.learnable_attention_matrix(vectors)
                dot_prod = vectors @ user_emb.unsqueeze(0).transpose(1, 0)
                sortmax_dot_prod = nn.functional.softmax(dot_prod, 0)
                pooled_vector = (sortmax_dot_prod * vectors).sum(axis=0)
            else:
                raise ValueError("Unsupported pooling type.")
        device = next(self.parameters()).device
        return pooled_vector.to(device)

    @property
    def embedding_size(self) -> int:
        """Function that return the output size of the model.

        Return: output_size (int): the output size of the model
        """
        return 2 * self.backbone_embd_size

    def set_pooling_type(self, pooling_type: str) -> None:
        """Function that change pooling type of the model."""
        if pooling_type not in ["mean", "max", "attention", "learnable_attention"]:
            raise ValueError("Unknown pooling type.")

        self.pooling_type = pooling_type
        if pooling_type == "learnable_attention":
            if self.learnable_attention_matrix is None:
                self.learnable_attention_matrix = nn.Linear(
                    self.backbone_embd_size, self.backbone_embd_size
                )
