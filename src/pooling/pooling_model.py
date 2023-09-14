import pandas as pd
import numpy as np

import torch


class PoolingModel(torch.nn.Module):
    def __init__(self,
                 train_dataloader,
                 local_model,
                 agregating_model: torch.nn.Module = None,
                 pooling_type: str = "mean",
                 local_model_emb_dim: int = None,
                 agregating_model_emb_dim: int = None
                 ) -> None:
        
        """Initialize method for PoolingModel

        Args: 
            local_model ():  Local embeding model
            argegating_model (torch.nn.Module): Model for agregating pooled embeddings
            train_dataloader (train_dataloader from CustomColesValidationDataset): Dataloader for calculating global 
                embedings from local sequences, train_batch_size need to be set equal to 1
            pooling_type (str): "max" or "mean", type of pooling
        """
        super().__init__()
        self.local_model = local_model
        self.device = local_model.device
        if agregating_model is not None:
            agregating_model.to(local_model.device)
        self.agregating_model = agregating_model
        self.local_model_emb_dim = local_model_emb_dim
        self.agregating_model_emb_dim = agregating_model_emb_dim
        self.pooled_embegings_dataset = self.make_pooled_embegings_dataset(train_dataloader, pooling_type)
        
    def make_pooled_embegings_dataset(self, train_dataloader, pooling_type):

        """Creation of pooled embeding dataset. This function for each timestamp get 
        sequences in dataset which ends close to this timestamp, 
        make local embedding out of them and pool them together

        Args: 
            train_dataloader (train_dataloader from CustomColesValidationDataset): Dataloader for calculating global 
                embedings from local sequences (train_batch_size need to be set equal to 1!)
            pooling_type (str): "max" or "mean", type of pooling

        Return:
            pooled_embegings_dataset(dict): dictionary containing timestamps and pooling vectors for that timestamps
        """
        data = {}
        for i, (x, y) in enumerate(train_dataloader):
            data[i] = {}

            x = x.to(self.device)
            embs = self.local_model(x).unsqueeze(1).detach().cpu().numpy()
            times = x.payload["event_time"][:, -1].cpu().numpy()
            for emb, time in zip(embs, times):
                    data[i][time] = emb
        
        data = pd.DataFrame(data).sort_index().ffill()
        pooled_embegings_dataset = {}
        for time in data.index:
            vectors = np.concatenate(data.loc[time].dropna().values)
            if pooling_type == "max":
                pooled_vector = torch.tensor(np.max(vectors, axis=0))
            elif pooling_type == "mean":
                pooled_vector = torch.tensor(np.mean(vectors, axis=0))
            else:
                raise ValueError("Unsupported pooling type.")
            pooled_embegings_dataset[time] = pooled_vector

        return pooled_embegings_dataset

    def forward(self, batch):
        """Forward method. 
        """
        batch_of_global_poolings = torch.tensor([]).to(self.device)
        for i, event_time_seq in enumerate(batch.payload["event_time"]):
            if self.agregating_model is None:
                local_pooled_emb = self.make_local_pooled_embedding(event_time_seq[-1].item())
                local_pooled_emb = local_pooled_emb.to(self.device)
                batch_of_global_poolings = torch.cat((batch_of_global_poolings, local_pooled_emb.unsqueeze(0)))
            else:
                seq_of_pooled_embs = torch.tensor([]).to(self.device)
                for time in event_time_seq:
                    local_pooled_emb = self.make_local_pooled_embedding(time.item()).to(self.device)
                    seq_of_pooled_embs = torch.cat((seq_of_pooled_embs, local_pooled_emb.unsqueeze(0)))
                batch_of_global_poolings = torch.cat((batch_of_global_poolings, self.agregating_model(seq_of_pooled_embs).unsqueeze(0)))

        batch_of_local_embedings = self.local_model(batch)
        return  torch.cat((batch_of_local_embedings, batch_of_global_poolings.to(batch_of_local_embedings.device)), dim = 1)
    
    def make_local_pooled_embedding(self, time):
        """Function that find the most close timestamp in self.pooled_embegings_dataset 
        and return the pooling vector at this timestamp.

        Args: 
            time (int): timepoint for which we are looking for pooling vector

        Return:
            pooled_vector (torch.tensor): pooling vector for given timepoint

        """
        times = list(self.pooled_embegings_dataset.keys())
        indexes = np.where(np.array(times) < time)[0]

        if len(indexes) == 0:
            pooled_vector =  torch.rand(self.local_model_emb_dim)
        else:
            closest_time = times[indexes[-1]]
            pooled_vector = self.pooled_embegings_dataset[closest_time]

        if self.agregating_model is not None:
            pooled_vector = pooled_vector.to(self.device)

        return pooled_vector
    
    def get_emb_dim(self):
        if self.agregating_model is None:
            return 2 * self.local_model_emb_dim
        else:
            return self.local_model_emb_dim + self.agregating_model_emb_dim

    def cuda(self):
        self.local_model.cuda()
        if self.agregating_model is not None:
            self.agregating_model.cpu()
        self.device =  self.local_model.device
        return self

    def cpu(self):
        self.local_model.cpu()
        if self.agregating_model is not None:
            self.agregating_model.cpu()
        self.device =  self.local_model.device
        return self

    def to(self, device):
        self.local_model.to(device)
        if self.agregating_model is not None:
            self.agregating_model.cpu()
        self.device =  self.local_model.device
        return self