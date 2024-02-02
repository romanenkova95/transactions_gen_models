"""Code from the EasyTPP repository: https://github.com/ant-research/EasyTemporalPointProcess."""

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

MINUS_INF = -1e3


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate masked attention.

    Args:
    ----
        query (torch.Tensor): qeury (matrix Q) for attention
        key (torch.Tensor):  key (matrix K) for attention
        value (torch.Tensor): value (matrix V) for attention
        mask (torch.Tensor, optional): mask to compute masked attention
        dropout (Callable, optional): nn.Dropout that is applied to the result

    Returns:
    -------
        Tuple[torch.Tensor]:
            * attention values
            * coefficients for attention, Softmax(Q @ K.T / sqrt(d))
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(
            mask > 0, MINUS_INF
        )  # use smaller constant here, 1e9 caused overflows
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention class."""

    def __init__(
        self,
        n_head: int,
        d_input: int,
        d_model: int,
        dropout: float = 0.1,
        output_linear: bool = False,
    ) -> None:
        """Initialize Custom multi-head attention class.

        Args:
        ----
            n_head (int): number of attention heads
            d_input (int): input dimensionality
            d_model (int): hidden (model) size; note that out_size=2*d_model
            dropout (float, optional): dropout probability
            output_linear (bool, optional): if True, add linear output layer
        """
        super(MultiHeadAttention, self).__init__()

        assert (
            d_model % n_head == 0
        ), "Model dimensionality must be a multiple of number of heads"

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.d_input = d_input
        self.d_model = d_model

        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(self.d_input, self.d_model) for _ in range(3)]
                + [
                    nn.Linear(self.d_model, self.d_model),
                ]
            )
        else:
            self.linears = nn.ModuleList(
                [nn.Linear(self.d_input, self.d_model) for _ in range(3)]
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        output_weight: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the module.

        Args:
        ----
            query (torch.Tensor): qeury (matrix Q) for attention
            key (torch.Tensor): key (matrix K) for attention
            value (torch.Tensor): value (matrix V) for attention
            mask (torch.Tensor): mask to compute masked attention
            output_weight (bool, optional): if True, return attention weights as well

        Returns:
        -------
            * torch.Tensor: output of the model
                or
            * Tuple[torch.Tensor]: output of the model and attention weights
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]

        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x


class SublayerConnection(nn.Module):
    """Custom model for residual connections."""

    # used for residual connection
    def __init__(self, d_model: int, dropout: float) -> None:
        """Initialize custom model for residual connections.

        Args:
        ----
            d_model (int): hidden (model) size
            dropout (float): dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        """Forward pass through the model.

        Args:
        ----
            x (torch.Tensor): input tensor
            sublayer (Callable): initialized torch.nn module to be applied

        Returns:
        -------
            torch.Tensor: output of the module
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer for A-NHP model."""

    def __init__(
        self,
        d_model: int,
        self_attn: Callable,
        feed_forward: Optional[Callable] = None,
        use_residual: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Initialize encoder layer for A-NHP model.

        Args:
        ----
            d_model (int): hidden (model) size
            self_attn (Callable): torch.nn module for attention computation
            feed_forward (Optional[Callable], optional): implemented torch.nn module for FF layer, if needed
            use_residual (bool, optional): if True, use residual connections
            dropout (float, optional): dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList(
                [SublayerConnection(d_model, dropout) for _ in range(2)]
            )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the module.

        Args:
        ----
            x (torch.Tensor): input tensor
            mask (torch.Tensor): boolean mask for attention computation

        Returns:
        -------
            torch.Tensor: output of the model
        """
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)
