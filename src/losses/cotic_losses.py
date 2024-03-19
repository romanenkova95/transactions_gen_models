"""The module with the COTIC losses."""

import math

import torch
from torch import nn

from src.nn.cotic_components.ccnn import CCNN

from .log_cosh_loss import LogCoshLoss


class CoticLoss(nn.Module):
    """Loss computation for COTIC."""

    def __init__(
        self,
        type_loss_coeff: float = 1,
        time_loss_coeff: float = 1,
        sim_size: int = 100,
        type_pad_value: int = 0,
        reductions: dict[str, str] = {
            "log_likelihood": "mean",
            "type": "sum",
            "time": "mean",
        },
    ) -> None:
        """Initialize CoticLoss.

        Args:
        ----
            type_loss_coeff (float): weighting coefficient for the type loss
            time_loss_coeff (float): weighting coefficient for the type loss
            sim_size (int): number of simulated timestamps between events for likelihood computation
            type_pad_value (int): padding value for event type (mcc)
            reductions (dict): dict with reductions used for all 3 losses
        """
        super().__init__()

        if reductions["log_likelihood"] not in ["mean", "sum"]:
            raise ValueError("log_likelihood reduction is not in ['mean', 'sum']")

        self.reductions = reductions

        self.type_loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=type_pad_value, reduction=self.reductions["type"]
        )

        self.return_time_loss_func = LogCoshLoss(self.reductions["time"])
        self.type_loss_coeff = type_loss_coeff
        self.time_loss_coeff = time_loss_coeff
        self.sim_size = sim_size

    @staticmethod
    def compute_event(
        type_lambda: torch.Tensor, non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-likelihood of events.

        Args:
        ----
            type_lambda (torch.Tensor): intesitities for events.
            non_pad_mask (torch.Tensor): boolean mask indicating non-padding events.

        Returns:
        -------
            'event term' of log-likelihood
        """
        # add 1e-9 in case some events have 0 likelihood
        type_lambda += math.pow(10, -9)
        type_lambda.masked_fill_(~non_pad_mask.bool(), 1.0)

        result = torch.log(type_lambda)
        return result

    @staticmethod
    def __add_sim_times(times: torch.Tensor, sim_size: int) -> torch.Tensor:
        """Take batch of times and events and adds sim_size auxiliar times.

        Args:
        ----
            times (torch.Tensor): event times since start, shape = (bs, max_len)
            sim_size (int): number of simulated timestamps between events for likelihood computation

        Returns:
        -------
            bos_full_times: torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1)
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
        self,
        model: CCNN,
        enc_output: torch.Tensor,
        event_time: torch.Tensor,
        non_pad_mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Compute log-likelihood of non-events, using Monte Carlo integration.

        Args:
        ----
            model (CCNN): backbone CCNN model (for intensity prediction)
            enc_output (torch.Tensor): embeddings produces by the core (continuous convolutional part of the model)
            event_time (torch.Tensor): event times (padded)
            non_pad_mask (torch.Tensor): boolean mask indicating non-padding timestamps
            num_samples (int): sample size for MC-integration

        Returns:
        -------
            approximated integral (non-event term of log-likelohood)
        """
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
        model: CCNN,
        enc_output: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log of the intensity and the integral.

        Args:
        ----
            model (nn.Module): CCNN backbone model.
            enc_output (torch.Tensor): embedding obtained by the CCNN model that is fed into final_list of layers for intensity computation.
            event_time (torch.Tensor): true event times.
            event_type (torch.Tensor): true event types.

        Returns:
        -------
            * event term of log-likelihood (sum)
            * non-event term of log-likelihood (integral)
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
            # type_mask,
            self.sim_size,
        )
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll

    def type_loss(self, prediction: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Compute event type prediction head loss.

        Args:
        ----
            prediction (torch.Tensor): predicted event types
            types (torch.Tensor): true event types

        Returns:
        -------
            event type loss function value (reduced)
        """
        truth = types[:, 1:]  # do not take the 1st value - it cannot be predicted
        prediction = prediction[:, :-1, :]

        loss = self.type_loss_func(prediction.transpose(1, 2), truth)

        loss = torch.mean(loss)
        return loss

    def time_loss(
        self,
        prediction: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor,
    ) -> float:
        """Compute return time prediction head loss.

        Args:
        ----
            prediction (torch.Tensor): predicted return times
            event_time (torch.Tensor): true event times
            event_type (torch.Tensor): true event types (to compute non-padding mask)

        Returns:
        -------
            return time loss function value (reduced)
        """
        prediction.squeeze_(-1)

        mask = event_type.ne(0)[:, 1:]

        true = event_time[:, 1:] - event_time[:, :-1]  # get return time target
        prediction = prediction[
            :, :-1
        ]  # do not take the last prediction, as there is no target for it

        return self.return_time_loss_func(true[mask], prediction[mask])

    def compute_loss(
        self,
        model: CCNN,
        inputs: tuple[torch.Tensor, torch.Tensor],
        outputs: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Compute composite loss for the COTIC model.

        Args:
        ----
            model (CCNN): CCNN backbone model that is being trained
            inputs (tuple of torch.Tensors): batch received from the dataloader
            outputs (tuple of torch.Tensors): model output in the form (encoded_output, (event_time_preds, return_time_preds))

        Returns:
        -------
            value of the loss fucntion
        """
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            model, outputs[0], inputs[0], inputs[1]
        )

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
