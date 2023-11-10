from typing import Union, Tuple
from abc import ABC

import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanAbsoluteError


class CoticMetrics(ABC):
    """COTIC metrics computing class."""
    def __init__(
        self,
        num_types: int,
        type_pad_value: int = 0,
    ) -> None:
        """Initalize CoticMetrcics.
        
        Args:
            num_types (int) - total number of event types in the dataset
            type_pad_value (int) - padding value for event types (0, by default)
        """

        #self.num_types = num_types
        #self.type_pad_value = type_pad_value

        self.return_time_metric = MeanAbsoluteError()
        self.event_type_metric = Accuracy(
            task="multiclass", num_classes=num_types, ignore_index=type_pad_value
        )

        self.clear_values()

    def clear_values(self):
        """Clears stored metrics values."""
        
        self.__return_time_target = torch.Tensor([])
        self.__event_type_target = torch.Tensor([])
        self.__return_time_preds = torch.Tensor([])
        self.__event_type_preds = torch.Tensor([])

    @staticmethod
    def get_return_time_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """Take input batch and returns the corresponding return time targets as 1d Tensor.

        Args:
            inputs (Tuple or torch.Tensor) - batch received from the dataloader

        Returns:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        event_time = inputs[0]
        return_time = event_time[:, 1:] - event_time[:, :-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time[mask]

    @staticmethod
    def get_event_type_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """Take input batch and returns the corresponding event type targets as 1d Tensor.

        Args:
            inputs (Tuple or torch.Tensor) - batch received from the dataloader

        Returns:
            event_type_target - torch.Tensor, 1d Tensor with event type targets
        """
        event_type = inputs[1][:, 1:]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type[mask]

    @staticmethod
    def get_return_time_predicted(
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Get return time predictions from model outputs.

        Args:
            inputs (Tuple or torch.Tensor) - batch received from the dataloader
            outputs (Tuple or torch.Tensor) - model output in the form (encoded_output, (event_time_preds, return_time_preds))

        Returns:
            return_time_predicted - torch.Tensor, 1d Tensor with return time prediction
        """
        return_time_prediction = outputs[1][0].squeeze_(-1)[:, 1:-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time_prediction[mask]

    @staticmethod
    def get_event_type_predicted(
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Get event type predictions from model outputs.

        Args:
            inputs (Tuple or torch.Tensor) - batch received from the dataloader
            outputs (Tuple or torch.Tensor) - model output in the form (encoded_output, (event_time_preds, return_time_preds))

        Returns:
            event_type_predicted - torch.Tensor, 2d Tensor with event type unnormalized predictions
        """
        event_type_prediction = outputs[1][1][:, 1:-1, :]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type_prediction[mask, :]

    def update(
        self, inputs: Union[Tuple, torch.Tensor], outputs: Union[Tuple, torch.Tensor]
    ) -> None:
        """Compute predictions and targets for a batch and store values.
        
        Args:
            inputs (Tuple or torch.Tensor) - batch received from the dataloader
            outputs (Tuple or torch.Tensor) - model output in the form (encoded_output, (event_time_preds, return_time_preds))
        """
        step_return_time_target = self.get_return_time_target(inputs)
        step_event_type_target = self.get_event_type_target(inputs)

        step_return_time_preds = self.get_return_time_predicted(inputs, outputs)
        step_event_type_preds = self.get_event_type_predicted(inputs, outputs)

        # store computed values
        self.__return_time_target = torch.concat(
            [
                self.__return_time_target,
                step_return_time_target.detach().clone().cpu(),
            ]
        )
        self.__event_type_target = torch.concat(
            [
                self.__event_type_target,
                step_event_type_target.detach().clone().cpu(),
            ]
        )
        self.__return_time_preds = torch.concat(
            [
                self.__return_time_preds,
                step_return_time_preds.detach().clone().cpu(),
            ]
        )
        self.__event_type_preds = torch.concat(
            [
                self.__event_type_preds,
                step_event_type_preds.detach().clone().cpu(),
            ]
        )

    def compute(self) -> Tuple[float, float, float]:
        """Compute metrics for the set of stored predictions and targets.
        
        Returns:
            return_time_metric (reduced)
            event_type_metric (reduced) 
        """
        return_time_metric = self.return_time_metric(
            self.__return_time_preds, self.__return_time_target
        )
        event_type_metric = self.event_type_metric(
            F.softmax(self.__event_type_preds, dim=1),
            self.__event_type_target,
        )

        # clear values, initialize new empty tensors for predictions and targets
        self.clear_values()

        return return_time_metric, event_type_metric
