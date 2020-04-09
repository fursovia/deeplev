from typing import List, Optional, Union, Iterable

from overrides import overrides

import numpy as np
import torch
from sklearn import metrics
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("recall_at_k")
class RecallAtK(Metric):
    def __init__(self, lower_k: Optional[int] = None, upper_k: int = 1):
        if upper_k <= 0:
            raise ConfigurationError("upper_k passed to Recall at K must be > 0")
        if lower_k is None:
            lower_k = upper_k
        elif upper_k < lower_k:
            raise ConfigurationError("upper_k passed to Recall at K must be >= lower_k")

        self._lower_k = lower_k
        self._upper_k = upper_k
    
    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ):
        if mask is None:
            mask = torch.ones(targets.shape[0]).bool()

        predictions, targets, mask = self.detach_tensors(predictions, targets, mask)

        self._recall = [_recall_at_k(predictions, targets, k) for k in range(self._lower_k, self._upper_k + 1)]       
        self._recall = (torch.Tensor(self._recall) * mask).T

    @overrides
    def get_metric(self, reset: bool = False):
        recall = self._recall
        if reset:
            self.reset()
        return recall
    
    @overrides
    def reset(self):
        self._upper_k = 1
        self._lower_k = 1
        self._recall = 0.0

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


def _recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int):
    return [np.intersect1d(x, y).shape[0] / len(x) for x, y in zip(predictions[:, :k], targets[:, :k])]
