"""Code from the EasyTPP repo: https://github.com/ant-research/EasyTemporalPointProcess."""

from typing import Tuple

import torch
import torch.nn as nn


class ContTimeLSTMCell(nn.Module):
    """LSTM Cell in Neural Hawkes Process, NeurIPS'17."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_event_types_pad: int,
        pad_token_id: int,
        beta: int = 1,
    ) -> None:
        """Initialize the continuous LSTM cell.

        Args:
        ----
            embed_dim (int): size of the event types embeddings
            hidden_dim (int): dimensionality of the hidden state
            num_event_types_pad (int): number of event types in the dataset (including padding)
            pad_token_id (int): event type number used as padding value
            beta (int): beta coefficient in nn.Softplus
        """
        super(ContTimeLSTMCell, self).__init__()

        self.layer_type_emb = nn.Embedding(
            num_event_types_pad, embed_dim, padding_idx=pad_token_id
        )

        self.init_dense_layer(embed_dim, hidden_dim, bias=True, beta=beta)

    def init_dense_layer(
        self, embed_dim: int, hidden_dim: int, bias: bool, beta: int
    ) -> None:
        """Initialize linear layers given Equations (5a-6c) in the paper.

        Args:
        ----
            embed_dim (int): size of the event types embeddings
            hidden_dim (int): dim of hidden state
            bias (bool): whether to use bias term in nn.Linear
            beta (float): beta coefficient in nn.Softplus
        """
        self.layer_input = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_forget = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_output = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_input_bar = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_forget_bar = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_pre_c = nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias)
        self.layer_decay = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim, bias=bias),
            nn.Softplus(beta=beta),
        )

    def forward(
        self,
        event_types_i: torch.Tensor,
        hidden_i_minus: torch.Tensor,
        cell_i_minus: torch.Tensor,
        cell_bar_i_minus_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the continuous-time LSTM cell.

        Args:
        ----
            event_types_i (tensor): event types vector at t_i.
            hidden_i_minus (tensor): hidden state at t_i-
            cell_i_minus (tensor): cell state at t_i-
            cell_bar_i_minus_1 (tensor): cell bar state at t_{i-1}

        Returns:
        -------
            a tuple of torch.Tensors:
                * cell state
                * cell bar state
                * decay
                * output at t_i
        """
        # event embeddings
        x_i = self.layer_type_emb(event_types_i)

        x_i_ = torch.cat((x_i, hidden_i_minus), dim=1)

        # update input gate - Equation (5a)
        gate_input = torch.nn.Sigmoid()(self.layer_input(x_i_))

        # update forget gate - Equation (5b)
        gate_forget = torch.nn.Sigmoid()(self.layer_forget(x_i_))

        # update output gate - Equation (5d)
        gate_output = torch.nn.Sigmoid()(self.layer_output(x_i_))

        # update input bar - similar to Equation (5a)
        gate_input_bar = torch.nn.Sigmoid()(self.layer_input_bar(x_i_))

        # update forget bar - similar to Equation (5b)
        gate_forget_bar = torch.nn.Sigmoid()(self.layer_forget_bar(x_i_))

        # update gate z - Equation (5c)
        gate_pre_c = torch.tanh(self.layer_pre_c(x_i_))

        # update gate decay - Equation (6c)
        gate_decay = self.layer_decay(x_i_)

        # update cell state to t_i+ - Equation (6a)
        cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c

        # update cell state bar - Equation (6b)
        cell_bar_i = gate_forget_bar * cell_bar_i_minus_1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(
        self,
        cell_i: torch.Tensor,
        cell_bar_i: torch.Tensor,
        gate_decay: torch.Tensor,
        gate_output: torch.Tensor,
        dtime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cell and hidden state decay according to Equation (7).

        Args:
        ----
            cell_i (tensor): cell state at t_i.
            cell_bar_i (tensor): cell bar state at t_i.
            gate_decay (tensor): gate decay state at t_i.
            gate_output (tensor): gate output state at t_i.
            dtime (tensor): delta time to decay.

        Returns:
        -------
            a tuple of torch.Tensors
            * cell state after the decay
            * hidden state after the decay
        """
        c_t = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(-gate_decay * dtime)
        h_t = gate_output * torch.tanh(c_t)

        return c_t, h_t
