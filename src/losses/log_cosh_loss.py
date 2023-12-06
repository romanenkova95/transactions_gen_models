import torch
import math


class LogCoshLoss(torch.nn.Module):
    """Custom LogCoshLoss for return time prediction head training."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("LogCoshLoss: reduction not in ['mean', 'sum']")
        self.reduction = reduction

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Compute LogCosh function."""
        return torch.where(
            torch.abs(x) > 10,
            torch.abs(x),
            x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)
        )

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        if self.reduction == "mean":
            return torch.mean(self._log_cosh(y_pred - y_true))
        else:
            return torch.sum(self._log_cosh(y_pred - y_true))