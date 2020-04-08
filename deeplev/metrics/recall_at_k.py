from typing import List, Optional, Union, Iterable

from overrides import overrides

import numpy as np
import torch
from sklearn import metrics
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("recall_at_k")
class RecallAtK(Metric):
    def __init__(self, top_k):
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Recall at K must be > 0")
        self._top_k = top_k
    
    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ):
        if mask is None:
            mask = torch.ones(targets.shape[0]).bool()
        
        predictions, targets, mask = self.detach_tensors(predictions[:, :self._top_k], targets[:, :self._top_k], mask)
        
        self._recall = torch.tensor([np.intersect1d(x, y).shape[0] / len(x) for x, y in zip(predictions, targets)])
        self._recall *= mask

    @overrides
    def get_metric(self, reset: bool = False):
        recall = self._recall
        if reset:
            self.reset()
        return recall
    
    @overrides
    def reset(self):
        self._top_k = 0
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
