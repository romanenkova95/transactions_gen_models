from typing import Optional
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import numpy as np

from ptls.frames.abs_module import ABSModule
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from torchmetrics import MeanMetric
from src.losses import HierarchicalContrastiveLoss

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def mask_input(x, mask):
    shape = x.size()

    if mask == 'binomial':
        mask = generate_binomial_mask(shape[0], shape[1]).to(x.device)
    elif mask == 'continuous':
        mask = generate_continuous_mask(shape[0], shape[1]).to(x.device)
    elif mask == 'all_true':
        mask = x.payload.new_full((shape[0], shape[1]), True, dtype=torch.bool)
    elif mask == 'all_false':
        mask = x.payload.new_full((shape[0], shape[1]), False, dtype=torch.bool)
    elif mask == 'mask_last':
        mask = x.payload.new_full((shape[0], shape[1]), True, dtype=torch.bool)
        mask[:, -1] = False

    x[~mask] = 0

    return x


class TS2Vec(ABSModule):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        encoder: DictConfig,
        optimizer_partial: DictConfig,
        lr_scheduler_partial: DictConfig,
        head: Optional[DictConfig] = None,
        loss: Optional[DictConfig] = None,
        col_time: str = "event_time",
        mask_mode: str = "binomial",        
    ) -> None:
        ''' Initialize TS2Vec module.
        
        Args:
        '''
        
        self.save_hyperparameters()
        enc: SeqEncoderContainer = instantiate(encoder)
        
        if head is None:
            head = Head(use_norm_encoder=True)
        else:
            head = instantiate(head)
        
        if loss is None:
            loss = HierarchicalContrastiveLoss(alpha=0.5, temporal_unit=0)
        else:
            loss = instantiate(loss)
            

        self.temporal_unit = loss.temporal_unit
        self.mask_mode = mask_mode
        
        super().__init__(
            seq_encoder=enc,
            loss=loss,
            optimizer_partial=instantiate(optimizer_partial, _partial_=True),
            lr_scheduler_partial=instantiate(lr_scheduler_partial, _partial_=True)
        )

        self.encoder = enc
        self._head = head
        self.valid_loss = MeanMetric()

        self.col_time = col_time

    def shared_step(self, x, y):
        trx_encoder = self._seq_encoder.trx_encoder
        seq_encoder = self._seq_encoder.seq_encoder 

        seq_lens = x.seq_lens
        t = x.payload[self.col_time]
        x = trx_encoder(x).payload

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        input1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        input2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        
        t = take_per_row(t, crop_offset + crop_eleft, crop_right - crop_eleft)
        t = t[:, -crop_l:]
        
        input1_masked = mask_input(input1, self.mask_mode)
        input2_masked = mask_input(input2, self.mask_mode)
        
        out1 = seq_encoder(PaddedBatch(input1_masked, seq_lens)).payload
        out1 = out1[:, -crop_l:]

        out2 = seq_encoder(PaddedBatch(input2_masked, seq_lens)).payload
        out2 = out2[:, :crop_l]
        
        if self._head is not None:
            out1 = self._head(out1)
            out2 = self._head(out2)

        return (out1, out2, t), y

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.valid_loss(loss)

    def validation_epoch_end(self, outputs):
        self.log(f'valid_loss', self.valid_loss, prog_bar=True)

    @property
    def is_requires_reduced_sequence(self):
        return False
    
    @property
    def metric_name(self):
        return "valid_loss"