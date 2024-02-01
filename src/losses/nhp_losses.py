"""Based on code from the EasyTPP repository: https://github.com/ant-research/EasyTemporalPointProcess."""

from typing import Tuple

import torch
import torch.nn as nn

from ..nn.nhp_seq_encoder import NHPEncoder
from ..nn.attn_nhp_seq_encoder import AttnNHPSeqEncoder


class NHPLoss(nn.Module):
    """Class for HNP loss computation."""
    def __init__(self, loss_integral_num_sample_per_step: int) -> None:
        """Initialize class for HNP loss computation. 

        Args:
            loss_integral_num_sample_per_step (int): number of samples for MC integral approximation 
        """
        super().__init__()
        
        self.loss_integral_num_sample_per_step = loss_integral_num_sample_per_step
        self.eps = torch.finfo(torch.float32).eps
        
    def compute_loglikelihood(
        self,
        time_delta_seq: torch.Tensor,
        lambda_at_event: torch.Tensor,
        lambdas_loss_samples: torch.Tensor,
        seq_mask: torch.Tensor,
        lambda_type_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
        ----
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at (right after) the event
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types], intensity at sampling times
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events
            lambda_type_mask (tensor): [batch_size, seq_len, num_event_types], type mask matrix to mask the padded event types

        Returns:
        -------
            A tuple of:
                * torch.Tensor: event loglikelihood
                * torch.Tensor: non-event loglikehood (integral approximation)
                * int: number of events
        """
        # Sum of lambda over every type and every event point
        # [batch_size, seq_len]
        event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=-1) + self.eps

        # mask the pad event
        event_lambdas = event_lambdas.masked_fill_(seq_mask == 0, 1.0)

        # [batch_size, seq_len)
        event_ll = torch.log(event_lambdas)

        # Compute the big lambda integral in equation (8) of NHP paper
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # [batch_size, seq_len, n_loss_sample]
        lambdas_total_samples = lambdas_loss_samples.sum(dim=-1)

        # interval_integral - [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        non_event_ll = lambdas_total_samples.mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        return event_ll, non_event_ll, num_events
    
    @staticmethod
    def make_dtime_loss_samples(time_delta_seq: torch.Tensor, loss_integral_num_sample_per_step: int) -> torch.Tensor:
        """Generate the time point samples for every interval.

        Args:
        ----
            time_delta_seq (tensor): [batch_size, seq_len], sequences of time differences between events
            loss_integral_num_sample_per_step (int): number of samples for MC integral approximation  

        Returns:
        -------
            tensor: [batch_size, seq_len, n_samples], tensor with sampled (auxiliary) times between events
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(
            start=0.0,
            end=1.0,
            steps=loss_integral_num_sample_per_step,
            device=time_delta_seq.device
        )[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes
    
    @staticmethod
    def compute_states_at_sample_times(
        seq_encoder: NHPEncoder, decay_states: torch.Tensor, sample_dtimes: torch.Tensor
    ) -> torch.Tensor:
        """Compute the states at sampling times.

        Args:
        ----
            seq_encoder (NHPEncoder): implemented NHP sequence encoder with .rnn_cell() method
            decay_states (tensor): states right after the events
            sample_dtimes (tensor): delta times in sampling

        Returns:
        -------
            tensor: hiddens states at sampling times
        """
        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = decay_states.unbind(dim=-2)

        # Use broadcasting to compute the decays at all time steps
        # at all sample points
        # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
        # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
        _, h_ts = seq_encoder.rnn_cell.decay(
            cells[:, :, None, :],
            cell_bars[:, :, None, :],
            decays[:, :, None, :],
            outputs[:, :, None, :],
            sample_dtimes[..., None]
        )

        return h_ts
        
    def compute_loss(
        self,
        seq_encoder: NHPEncoder,
        time_seqs: torch.Tensor, # for interface consistency only
        time_delta_seqs: torch.Tensor, 
        type_seqs: torch.Tensor, 
        batch_non_pad_mask: torch.Tensor,
        attention_mask: torch.Tensor, # for interface consistency only
        type_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-likelihood loss for the NHP model.

        Args:
        ----
            seq_encoder (NHPEncoder): implemented NHP sequence encoder with .rnn_cell() method
            time_seqs (torch.Tensor): times of events (batch)
            type_seqs (torch.Tensor): types of events (batch)
            batch_non_pad_mask (torch.Tensor): boolean mask indicating non-padding events
            attention_mask (torch.Tensor): bollean mask for masked attention computation

        Returns:
        -------
            torch.Tensor: value of the loss function
        """
        hiddens_ti, decay_states = seq_encoder.run_batch(time_delta_seqs, type_seqs)

        # Num of samples in each batch and num of event time point in the sequence
        # batch_size, seq_len, _ = hiddens_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = seq_encoder.layer_intensity(hiddens_ti)

        # Compute the big lambda integral in Equation (8)
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:], self.loss_integral_num_sample_per_step)

        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        state_t_sample = self.compute_states_at_sample_times(seq_encoder, decay_states, interval_t_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = seq_encoder.layer_intensity(state_t_sample)
        
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            lambda_type_mask=type_mask[:, 1:]
        )

        loss = - (event_ll - non_event_ll).sum()
        return loss / num_events
    
class AttnNHPLoss(NHPLoss):
    """Class for A-HNP loss computation."""
    @staticmethod
    def compute_states_at_sample_times(
        seq_encoder: AttnNHPSeqEncoder,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_times: torch.Tensor
    ) -> torch.Tensor:
        """Compute the states at sampling times.

        Args:
        ----
            seq_encoder (AttnNHPSeqEncoder): implemented AttnNHP sequence encoder
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times
            type_seqs (tensor): [batch_size, seq_len], sequences of event types
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for attention computation
            sample_dtimes (tensor): delta times in sampling

        Returns:
        ------
            tensor: hiddens states at sampling times
        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_times.size(-1)

        # [num_samples, batch_size, seq_len]
        sample_times = sample_times.permute((2, 0, 1))
        # [num_samples * batch_size, seq_len]
        _sample_time = sample_times.reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _types = type_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _times = time_seqs.expand(num_samples, -1, -1).reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _attn_mask = attention_mask.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * batch_size,
                                                                                         seq_len,
                                                                                         seq_len)
        # [num_samples * batch_size, seq_len, hidden_size]
        inputs = (_times, _types, _attn_mask, _sample_time)
        encoder_output = seq_encoder(inputs)

        # [num_samples, batch_size, seq_len, hidden_size]
        encoder_output = encoder_output.reshape(num_samples, batch_size, seq_len, -1)
        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = encoder_output.permute((1, 2, 0, 3))
        return encoder_output
    
    def compute_intensities_at_sample_times(
        self,
        seq_encoder: AttnNHPSeqEncoder,
        time_seqs: torch.Tensor, 
        type_seqs: torch.Tensor, 
        sample_times: torch.Tensor, 
        attention_mask: torch.Tensor, 
        compute_last_step_only: bool = False 
    ) -> torch.Tensor:
        """Compute the intensity at sampled times.

        Args:
        ----
            seq_encoder (AttnNHPSeqEncoder): implemented AttnNHP sequence encoder
            time_seqs (tensor): sequences of timestamps
            type_seqs (tensor): sequences of event types
            sample_times (tensor): sampled time delta sequence
            attention_mask (tensor): boolean mask for masked attention computation
            compute_last_step_only (bool): if True, compute only intensity value at the last event

        Returns:
        -------
            tensor: intensities at sampled_times, [batch_size, seq_len, num_samples, event_num]
        """
        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        if sample_times.size()[1] < time_seqs.size()[1]:
            # we pass sample_dtimes for last time step here
            # we do a temp solution
            # [batch_size, seq_len, num_samples]
            sample_times = time_seqs[:, :, None] + torch.tile(sample_times, [1, time_seqs.size()[1], 1])

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(
            seq_encoder, time_seqs, type_seqs, attention_mask, sample_times
        )

        if compute_last_step_only:
            lambdas = seq_encoder.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = seq_encoder.layer_intensity(encoder_output)
        return lambdas
    
    def compute_loss(
        self,
        seq_encoder: AttnNHPSeqEncoder,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        batch_non_pad_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        type_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-likelihood loss for the A-NHP model.

        Args:
        ----
            seq_encoder (AttnNHPSeqEncoder): implemented AttnNHP sequence encoder
            time_seqs (torch.Tensor): sequences of timestamps
            time_delta_seqs (torch.Tensor): sequences of time differences
            type_seqs (torch.Tensor): sequences of event types
            batch_non_pad_mask (torch.Tensor): boolean mask indicating non-padding events
            attention_mask (torch.Tensor): boolean mask for masked attention computation
            type_mask (torch.Tensor): type mask matrix to mask the padded event types

        Returns:
        -------
            torch.Tensor: value of the loss function
        """
        # 1. compute event-loglikelihood
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        
        enc_out = seq_encoder.run_batch(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, 1:, :-1], time_seqs[:, 1:])
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = seq_encoder.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(time_delta_seqs[:, 1:], self.loss_integral_num_sample_per_step)

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + time_seqs[:, :-1].unsqueeze(-1)

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(
            seq_encoder, 
            time_seqs[:, :-1],
            type_seqs[:, :-1],
            sample_times,
            attention_mask[:, 1:, :-1]
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        loss = - (event_ll - non_event_ll).sum()
        return loss / num_events