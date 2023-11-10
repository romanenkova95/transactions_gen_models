import torch
import math


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss for return time prediction head training, approximates MAE"""
    def __init__(self, reduction: str = "mean") -> None:
        """Initialize LogCoshLoss.
        
        Args:
            reduction (str) - type of loss reduction ('mean' or 'sum')
        """
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("LogCoshLoss: reduction not in ['mean', 'sum']")
        self.__reduction = reduction

    @staticmethod
    def __log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Compute LogCosh function.
        
        Args:
            x (torch.Tensor) - input values
        
        Returns:
            LogCosh(x) for |x| < 10, and |x| elsewhere
        """
        out = x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)
        out[torch.abs(x) > 10] = torch.abs(x[torch.abs(x) > 10])
        return out

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute loss, apply reduction.
        
        Args:
            y_true (torch.Tensor) - true return times
            y_pred (torch.Tensor) - predicted return times
        
        Returns:
            value of the loss function for a batch
        """
        if self.__reduction == "mean":
            return torch.mean(self.__log_cosh(y_pred - y_true))
        else:
            return torch.sum(self.__log_cosh(y_pred - y_true))
