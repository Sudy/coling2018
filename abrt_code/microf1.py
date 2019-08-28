from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
# from ignite._utils import to_onehot
from sklearn.metrics import f1_score

class Microf1(Metric):
  def __init__(self, average=False, output_transform=lambda x: x):
    super(Microf1, self).__init__(output_transform)
    # self._average = average
    self.pred_label, self.gold_label = [], []

  def reset(self):
    self.pred_label = []
    self.gold_label = []

  def update(self, output):
    #y_pred (batch_size, num_classes)
    #y (batch_size)
    y_pred, y = output

    indices = torch.max(y_pred, 1)[1]
    
    self.pred_label.extend(indices.tolist())
    self.gold_label.extend(y.tolist())

  def compute(self):
    if self.pred_label is None or self.gold_label is None:
      raise NotComputableError('Microf1 must have at least one example before it can be computed')
    return f1_score(self.gold_label, self.pred_label, average = 'micro')
