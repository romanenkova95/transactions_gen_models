import math

import torch
import torch.nn as nn

from typing import Tuple, Union

from .log_cosh_loss import LogCoshLoss


class CoticLoss(nn.Module):
    def __init__(
        self,
        type_loss_coeff: float = 1,
        time_loss_coeff: float = 1,
        sim_size: int = 100,
        type_pad_value: int = 0,
        reductions: dict = {"log_likelihood": "mean", "type": "sum", "time": "mean"},
    ):
        super().__init__()

        self.reductions = reductions

        self.type_loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=type_pad_value, reduction=self.reductions["type"]
        )

        self.return_time_loss_func = LogCoshLoss(self.reductions["time"])
        self.type_loss_coeff = type_loss_coeff
        self.time_loss_coeff = time_loss_coeff
        self.sim_size = sim_size

    @staticmethod
    def compute_event(type_lambda, non_pad_mask):
        """Log-likelihood of events."""

        # add 1e-9 in case some events have 0 likelihood
        type_lambda += math.pow(10, -9)
        type_lambda.masked_fill_(~non_pad_mask.bool(), 1.0)

        result = torch.log(type_lambda)
        return result

    @staticmethod
    def __add_sim_times(
        times: torch.Tensor, sim_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes batch of times and events and adds sim_size auxiliar times

        args:
            times  - torch.Tensor of shape (bs, max_len), that represents event arrival time since start
            featuers - torch.Tensor of shape (bs, max_len, d), that represents event features
            sim_size - int, number of points to simulate

        returns:
            bos_full_times - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1)
                         that consists of times and sim_size auxiliar times between events
        """
        delta_times = times[:, 1:] - times[:, :-1]
        sim_delta_times = (
            (
                torch.rand(list(delta_times.shape) + [sim_size]).to(times.device)
                * delta_times.unsqueeze(2)
            )
            .sort(dim=2)
            .values
        )
        full_times = torch.concat(
            [sim_delta_times.to(times.device), delta_times.unsqueeze(2)], dim=2
        )
        full_times = full_times + times[:, :-1].unsqueeze(2)
        full_times[delta_times < 0, :] = 0
        full_times = full_times.flatten(1)
        bos_full_times = torch.concat(
            [torch.zeros(times.shape[0], 1).to(times.device), full_times], dim=1
        )

        return bos_full_times

    def compute_integral_unbiased(
        self, model, enc_output, event_time, non_pad_mask, type_mask, num_samples
    ):
        """Log-likelihood of non-events, using Monte Carlo integration."""

        bos_full_times = self.__add_sim_times(event_time, num_samples)
        all_lambda = model.final(
            bos_full_times, event_time, enc_output, non_pad_mask.bool(), num_samples
        )  # shape = (bs, (num_samples + 1) * L + 1, num_types)

        bs, _, num_types = all_lambda.shape

        between_lambda = (
            all_lambda.transpose(1, 2)[:, :, 1:]
            .reshape(bs, num_types, event_time.shape[1] - 1, num_samples + 1)[..., :-1]
            .transpose(1, 2)
        )

        diff_time = (event_time[:, 1:] - event_time[:, :-1]) * non_pad_mask[:, 1:]
        between_lambda = torch.sum(between_lambda, dim=(2, 3)) / num_samples

        unbiased_integral = between_lambda * diff_time
        return unbiased_integral

    def event_and_non_event_log_likelihood(
        self,
        model: nn.Module,
        enc_output: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log of the intensity and the integral

        model -- CCNN model
        """

        non_pad_mask = event_type.ne(0).type(torch.float)

        type_mask = torch.zeros(
            [*event_type.size(), model.num_types], device=enc_output.device
        )
        for i in range(model.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(enc_output.device)

        event_time = torch.concat(
            [torch.zeros(event_time.shape[0], 1).to(event_time.device), event_time],
            dim=1,
        )
        non_pad_mask = torch.concat(
            [torch.ones(event_time.shape[0], 1).to(event_time.device), non_pad_mask],
            dim=1,
        ).long()
        all_lambda = model.final(
            event_time, event_time, enc_output, non_pad_mask.bool(), 0
        )

        type_lambda = torch.sum(
            all_lambda[:, 1:, :] * type_mask, dim=2
        )  # shape = (bs, L)

        # event log-likelihood
        event_ll = self.compute_event(type_lambda, non_pad_mask[:, 1:])
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, MC integration
        non_event_ll = self.compute_integral_unbiased(
            model,
            enc_output,
            event_time,
            non_pad_mask,
            type_mask,
            self.sim_size,
        )
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll

    def type_loss(self, prediction, types):
        """Event prediction loss, cross entropy."""

        # ??? convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
        truth = types[:, 1:]  # - 1
        prediction = prediction[:, :-1, :]

        loss = self.type_loss_func(prediction.transpose(1, 2), truth)

        loss = torch.mean(loss)
        return loss

    def time_loss(self, prediction, event_time, event_type):
        """Time prediction loss."""

        prediction.squeeze_(-1)

        mask = event_type.ne(0)[:, 1:]

        true = event_time[:, 1:] - event_time[:, :-1]
        prediction = prediction[:, :-1]

        return self.return_time_loss_func(true[mask], prediction[mask])

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding loss for backpropagation,
        one can use self.step_[return_time_target/event_type_target/return_time_predicted/event_type_predicted/ll_per_event] if needed

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            loss - torch.Tensor, loss for backpropagation
        """
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            model, outputs[0], inputs[0], inputs[1]
        )

        if self.reductions["log_likelihood"] not in ["mean", "sum"]:
            raise ValueError("log_likelihood reduction not in 'mean', 'sum'")

        if self.reductions["log_likelihood"] == "mean":
            ll_loss = -torch.mean(event_ll - non_event_ll)
        else:
            ll_loss = -torch.sum(event_ll - non_event_ll)

        type_loss = self.type_loss(outputs[1][1][:, 1:], inputs[1])
        time_loss = self.time_loss(outputs[1][0][:, 1:], inputs[0], inputs[1])

        return (
            ll_loss,
            self.type_loss_coeff * type_loss,  # return scaled loss components
            self.time_loss_coeff * time_loss,
        )
