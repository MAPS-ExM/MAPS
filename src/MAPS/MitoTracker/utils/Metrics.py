import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List


class Metrics:
    def __init__(self, metrics: str = '', labels_to_eval: List[int] = [1,2,3,4]) -> None:
        """
        Takes a string with different metrics separated by white spaces like
        'iou dice accuracy' and checks that corresponding methods are
        implemented to caluclate it based on boolean tensors which are called like
        'comp_{metric}_bool'.
        """
        requested_metrics = metrics.lower().split(' ')
        # Dictionary _metrics = {metric: [value1, value2, ..]} that stores for each requested
        # metric a bunch of values which can later be summarised with .summarise_metrics()
        # This allows to compute the metrics over several batches and then summarise to a single value
        self._metrics = {}  
        self.labels_to_eval = labels_to_eval
        for metric in requested_metrics:
            assert f'comp_{metric}_bool' in dir(self), \
                f'Method to compute {metric} is not implemented yet!'
            self._metrics[metric] = []

    def comp_metrics_from_bool(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Takes two boolean tensors and computes the metrics in self.metrics for them
        """
        for metric in self._metrics.keys():
            current_metric = getattr(self, f'comp_{metric}_bool')(pred, target).cpu().item()
            self._metrics[metric].append(current_metric)

    def comp_metrics(self, pred: torch.Tensor, target: torch.Tensor, ignore_labels: List[int] = [0], 
                     error_tol: int = 0, respect_ambiguity: bool = False) -> None:
        """
        Takes a multiclass prediction and computes for every label in self.labels_to_eval the specified metrics.
        These metrics then get averaged over the different labels and saved.
        The labels in <ignore_labels> (like the label 0 for the background) are not considered and skipped 
        """     
        for metric in self._metrics.keys():
            current_values = []
            for label in self.labels_to_eval:
                if label not in ignore_labels:
                    bool_pred = pred==label
                    bool_target = target==label
                    
                    if respect_ambiguity and label in [2,3]:
                        # If the model predicts matrix, it is okay if the GT is matrix or matrix/cristae. Same for cristae
                        # But we are strict: If the model predicts ambiguous, the target needs to be ambiguous as well
                        # This implies, that the model cant get away with just predicting ambiguous everywhere.
                        mask = torch.logical_and(bool_pred, target==4)  # Find area that is predicted matrix, but ambiguous in the GT
                        bool_target = torch.logical_or(bool_target, mask)  # Add that area to the target
                    
                    if error_tol > 0:
                        bool_pred = self.include_error_margin(bool_pred, bool_target, error_tol=error_tol)
                        
                    current_values.append(getattr(self, f'comp_{metric}_bool')(bool_pred, bool_target).cpu().item())
            
            self._metrics[metric].append(np.mean(current_values))

    def get(self, metrics: str) -> Union[float, List[float]]:
        """
        Splits the metrics string by white spaces and returns the current value
        If there is only a single metric requested, it just returns that float
        """
        res = []
        for metric in metrics.lower().split(' '):
            res.append(self._metrics[metric])
        if len(res) == 1:
            return res[0]
        else:
            return res

    def get_last(self, metrics: str) -> Union[float, List[float]]:
        """
        Splits the metrics string by white spaces and returns the most recent value computed
        If there is only a single metric requested, it just returns that float
        """
        res = []
        for metric in metrics.lower().split(' '):
            res.append(self._metrics[metric][-1])
        if len(res) == 1:
            return res[0]
        else:
            return res

    def summarise_metrics(self) -> None:
        """
        Replace the list of computed metrics with its mean
        """
        for metric in self._metrics.keys():
            self._metrics[metric] = np.mean(self._metrics[metric])

    def add_metric_tensorboard(self, writer: "SummaryWriter", iteration: int,
                               prefix: str = "", suffix: str = "") -> None:
        """
        Add all the metrics to the summary writer.
        """
        for metric in self._metrics.keys():
            label = prefix + self._transform_metric_label(metric) + suffix
            assert isinstance(self._metrics[metric], float), f'{metric} is not a float!'
            writer.add_scalar(label, self._metrics[metric], iteration)

    def _transform_metric_label(self, label: str) -> str:
        if label == 'iou':
            return 'IoU'
        else:
            return label.capitalize()

    @staticmethod
    def include_error_margin(pred: torch.Tensor, target: torch.Tensor, error_tol: int = 0):
        """
        Replaces the boundary of the prediction with the target to allow for an error margin. 
        """
        if not error_tol in [0,1]:
            raise NotImplementedError("Error tolerance only implemented for 0 and 1")
        
        # Need to insert a channel dimension
        # dilated = F.conv3d(pred[:, None].float(), weight=torch.ones((1,1,3,3,3)).to(pred.device), padding=1) > 0
        # eroded = F.conv3d(torch.logical_not(pred)[:, None].float(),  weight=torch.ones((1,1,3,3,3)).to(pred.device), padding=1) > 0
        
        # Lets only use a 2 d kernel effecitvely (a for loop might actually be more efficient, dont' know)
        kernel_size = 3
        kernel = torch.ones((1,1,kernel_size,kernel_size,kernel_size)).to(pred.device)
        kernel[:,:, 0] =0
        kernel[:,:, 2] =0

        dilated = F.conv3d(pred.float(), weight=kernel, padding=1) > 0
        eroded = torch.logical_not(F.conv3d(torch.logical_not(pred).float(),  weight=kernel, padding=1) > 0)
        boundary = torch.logical_xor(dilated, eroded)[:,0]  # Remove channel dimension again

        # pred[boundary] = target[boundary]
        pred[boundary[:,None]] = target[boundary[:,None]]

        return pred

    @staticmethod
    def _check_and_flatten(pred, target):
        assert pred.shape == target.shape, f"Shapes don't match! Pred: {pred.shape} vs Target: {target.shape}"
        if len(pred.shape) > 1:
            pred = pred.view(pred.shape[0], -1)
            target = target.view(target.shape[0], -1)
        else:
            pred, target = pred[None], target[None]
        return pred, target

    @staticmethod
    def comp_iou_bool(pred, target):
        # Compute the IoU per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        inter = torch.sum(torch.logical_and(pred, target), dim=1) 
        union = torch.sum(torch.logical_or(pred, target), dim=1)
        iou = inter / (union + 1e-8)
        return torch.mean(iou)

    @staticmethod
    def comp_dice_bool(pred, target):
        # Compute the dice per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        inter = torch.sum(torch.logical_and(pred, target), dim=1)
        denom = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
        dice = 2 * inter / (denom + 1e-8)
        return torch.mean(dice)

    @staticmethod
    def comp_accuracy_bool(pred, target):
        # Compute the accuracy per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        overlap = torch.sum(pred == target, dim=1)
        accuracy = overlap / target.shape[1]
        return torch.mean(accuracy)

    @staticmethod
    def comp_precision_bool(pred, target):
        # Compute the precision per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        true_positive = torch.logical_and(pred, target).sum(dim=1)
        pred_positive = pred.sum(dim=1)
        precision = true_positive / pred_positive
        return torch.mean(precision)

    @staticmethod
    def comp_recall_bool(pred, target):
        # Compute the recall per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        true_positive = torch.logical_and(pred, target).sum(dim=1)
        positive = target.sum(dim=1)
        recall = true_positive / positive
        return torch.mean(recall)