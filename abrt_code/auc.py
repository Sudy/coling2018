from __future__ import division

import torch
import numpy as np


from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot
from sklearn.metrics import average_precision_score

class AUC(Metric):
  """
  Calculates AUC.

  `update` must receive output of the form (y_pred, y).

  If `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the AUC for each class.
  """
  def __init__(self, average=False, output_transform=lambda x: x):
    super(AUC, self).__init__(output_transform)
    self._average = average

  def reset(self):
    self._actual = []
    self._pred = []

  def update(self, output):
    #y_pred (batch_size, num_classes)
    #y (batch_size)
    y_pred, y = output
    num_classes = y_pred.size(1)
    y_pred_ = torch.exp(y_pred)
    actual_onehot = to_onehot(y, num_classes)
    
    # if  self._actual is None or self._pred is None:
    self._actual.extend(actual_onehot.tolist())
    self._pred.extend(y_pred_.tolist())

  def compute(self):
    if self._actual is None or self._pred is None:
      raise NotComputableError('AUC must have at least one example before it can be computed')

    actual = np.asarray(self._actual)
    pred = np.asarray(self._pred)
    # print(pred)

    scores = [average_precision_score(\
      actual[:, i], pred[:, i]) for i in range(pred.shape[-1])]

    return scores