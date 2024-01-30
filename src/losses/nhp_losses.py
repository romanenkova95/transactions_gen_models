import torch
import torch.nn as nn

class NHPLoss(nn.Module):
    """Loss computation for HNP. """

    def __init__(self, loss_integral_num_sample_per_step) -> None:
        """Initialize .

        Args:
        ----
        """
        super().__init__()
        
        self.loss_integral_num_sample_per_step = loss_integral_num_sample_per_step
        self.eps = torch.finfo(torch.float32).eps
        
    def compute_loglikelihood(
        self,
        time_delta_seq,
        lambda_at_event,
        lambdas_loss_samples,
        seq_mask,
        lambda_type_mask
    ):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types], intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            lambda_type_mask (tensor): [batch_size, seq_len, num_event_types], type mask matrix to mask the padded event types.

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
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
    
    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(
            start=0.0,
            end=1.0,
            steps=self.loss_integral_num_sample_per_step,
            device=time_delta_seq.device
        )[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes
    
    @staticmethod
    def compute_states_at_sample_times(seq_encoder, decay_states, sample_dtimes): # Q: encoder or just rnn_cell ???
        """Compute the states at sampling times.

        Args:
            decay_states (tensor): states right after the events.
            sample_dtimes (tensor): delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
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
        
    def compute_loss(self, seq_encoder, time_delta_seqs, type_seqs, batch_non_pad_mask, type_mask):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """        
        inputs = (time_delta_seqs, type_seqs)
        hiddens_ti, decay_states = seq_encoder(inputs)

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
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

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

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        
        return loss / num_events