# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import PrecsionRecallF1score


def test_metric_init_assertion():
    with pytest.raises(
            AssertionError, match='`num_classes` is necessary'):
        PrecsionRecallF1score(task='multilabel', num_classes=None)
    with pytest.raises(
            AssertionError, match="task `'multilabel'` only supports"):
        PrecsionRecallF1score(task='multilabel', thrs=(0., 0.1, 0.4))
    with pytest.raises(
            ValueError, match='Expected argument `task` to either be'):
        PrecsionRecallF1score(task='threeclasses')


def test_metric_interface():
    preds = np.array([2, 0, 1, 1])
    labels = np.array([2, 1, 2, 0])

    # test predictions with labels
    metric = PrecsionRecallF1score(task='singlelabel', num_classes=3)
    assert isinstance(metric, BaseMetric)
    results = metric(preds, labels)
    assert isinstance(results, dict)

    # test predictions with pred_scores
    metric = PrecsionRecallF1score(task='multilabel', num_classes=3)
    assert isinstance(metric, BaseMetric)
    results = metric(preds, labels)
    assert isinstance(results, dict)
