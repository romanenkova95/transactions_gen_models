"""Losses for the TS2Vec module."""

import torch
import torch.nn.functional as F
from torch import nn


class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical Contrastive Loss for TS2Vec model."""

    def __init__(self, alpha: float, temporal_unit: int) -> None:
        """Initlize HierarchicalContrastiveLoss.

        Args:
        ----
            alpha (float): weighting coefficient
            temporal_unit (int): start computing temporal component after this level of hierarchy
        """
        super().__init__()

        self.alpha = alpha
        self.temporal_unit = temporal_unit

    @staticmethod
    def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute instance-wise component of contrastive loss.

        Args:
        ----
            z1 (torch.Tensor): embedding of the 1st augmented window
            z2 (torch.Tensor): embedding of the 2nd augmented window

        Returns:
        -------
            instance-wise component of the loss function
        """
        B = z1.size(0)
        if B == 1:
            return z1.new_tensor(0.0, requires_grad=True)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    @staticmethod
    def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute temporal component of contrastive loss.

        Args:
        ----
            z1 (torch.Tensor): embedding of the 1st augmented window
            z2 (torch.Tensor): embedding of the 1st augmented window

        Returns:
        -------
            temporal component of the loss function
        """
        T = z1.size(1)
        if T == 1:
            return z1.new_tensor(0.0, requires_grad=True)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T - 1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def hierarchical_contrastive_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, alpha: float, temporal_unit: int
    ) -> torch.Tensor:
        """Compute hierarchical contrastive loss for TS2Vec model.

        Args:
        ----
            z1 (torch.Tensor): embedding of the 1st augmented window
            z2 (torch.Tensor): embedding of the 2nd augmented window
            alpha (float): weighting coefficient
            temporal_unit (int): start computing temporal component after this level of hierarchy

        Returns:
        -------
            value of the loss function
        """
        loss = torch.tensor(0.0, device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            d += 1
        return loss / d

    def forward(
        self, embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        """Compute value of the loss function.

        Args:
        ----
            embeddings (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): embeddings of 2 windows and time, i.e. output of TS2Vec.shared_step()

        Returns:
        -------
            value of the loss function
        """
        out1, out2, _ = embeddings
        return self.hierarchical_contrastive_loss(
            out1, out2, self.alpha, self.temporal_unit
        )
