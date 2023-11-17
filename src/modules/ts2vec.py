from typing import Optional, Union, Tuple
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

def generate_continuous_mask(
    B: int, T: int, n: Union[int, float] = 5, l: Union[int, float] = 0.1
) -> torch.Tensor:
    """Generate continuous mask for time window augmentation.
    
    Args:
        B (int) - batch size
        T (int) - series lengths (num of timestamps)
        n (int or float) - number (or share) of steps for masking
        l (int or float) - number (or share) of maximum masking length
    
    Returns:
        torch.Tensor with boolean mask for sequences augmentation 
    """
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


def generate_binomial_mask(B: int, T: int, p: float = 0.5) -> torch.Tensor:
    """Generate binomial mask for time window augmentation.
    
    Args:
        B (int) - batch size
        T (int) - series lengths (num of timestamps) 
        p (float) - masking probability
        
    Returns:
        torch.Tensor with boolean mask for sequences augmentation 
    """
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def take_per_row(inputs: torch.Tensor, indx: np.array, num_elem: int) -> torch.Tensor:
    """Takes 'num_elem' from each row of 'A', starting from the indices provided in the 'indx'.
    
    Args:
        inputs (torch.Tensor) - original tensor with data
        indx (np.array) - array of indices to be taken
        num_elem (int) - number of element to be taken from each row
        
    Returns:
        subset of initial sequences data
    """
    all_indx = indx[:, None] + np.arange(num_elem)
    return inputs[torch.arange(all_indx.shape[0])[:, None], all_indx]


def mask_input(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask input using the given boolean mask.
    
    Args:
        inputs (torch.Tensor) - input sequences
        mask (torch.Tensor) - boolean mask
    
    Returns:
        masked input sequences
    """
    shape = inputs.size()

    if mask == 'binomial':
        mask = generate_binomial_mask(shape[0], shape[1]).to(inputs.device)
    elif mask == 'continuous':
        mask = generate_continuous_mask(shape[0], shape[1]).to(inputs.device)
    elif mask == 'all_true':
        mask = inputs.new_full((shape[0], shape[1]), True, dtype=torch.bool)
    elif mask == 'all_false':
        mask = inputs.new_full((shape[0], shape[1]), False, dtype=torch.bool)
    elif mask == 'mask_last':
        mask = inputs.new_full((shape[0], shape[1]), True, dtype=torch.bool)
        mask[:, -1] = False

    inputs[~mask] = 0

    return inputs


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
        mask_mode: str = "binomial" 
    ) -> None:
        ''' Initialize TS2Vec module.
        
        Args:
            encoder (DictConfig) - config for TS2Vec sequence encoder instantiation
            optimizer_partial (DictConfig) - config for optimizer instantiation (ptls format)
            lr_scheduler_partial (DictConfig) - config for lr scheduler instantiation (ptls format)
            head (DictConfig or None) - if not None, use this head after the backbone model,
                                        else use ptls.nn.head.Head(use_norm_encoder=True) 
            loss: (DictConfig or None) - if not None, contains DictConfig for TS2Vec loss instantiation, else use default loss,
                                         else use src.losses.HierarchicalContrastiveLoss with default hyper-parameters
            col_time (str) - name of the column containing timestamps
            mask_mode (str) - type of mask to be generated & used
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

    def shared_step(
        self, x: PaddedBatch, y: Optional[torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        """Shared training/validation/testing step.
        
        Args:
            x (PaddedBatch) - input sequences in ptls format
            y (torch.Tensor) - labels as provided by the dataloader
            
        Returns a tuple of:
            * (embedding of the 1st augmented window, embedding of the 2nd augmented window, timestamps)
            * original labels as provided by the dataloader 
        """
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

    def validation_step(self, batch: Tuple[PaddedBatch, Optional[torch.Tensor]], _) -> None:
        """Validation step of the model.
        
        Args:
            batch (Tuple[PaddedBatch, torch.Tensor]) padded batch that is fed into TS2Vec sequence encoder and labels
        """
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.valid_loss(loss)

    def validation_epoch_end(self, _) -> None:
        """Log loss for a validation epoch."""
        self.log(f'valid_loss', self.valid_loss, prog_bar=True)

    @property
    def is_requires_reduced_sequence(self) -> bool:
        """TS2Vec does not reduce sequence by default."""
        return False
    
    @property
    def metric_name(self) -> str:
        """Validation monitoring metric name."""
        return "valid_loss"