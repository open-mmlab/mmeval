# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest
import unittest
from distutils.version import LooseVersion

from mmeval.metrics import F1Score
from mmeval.utils import try_import

torch = try_import('torch')
flow = try_import('oneflow')


def test_init():
    with pytest.raises(AssertionError):
        F1Score(num_classes='3')

    with pytest.raises(AssertionError):
        F1Score(num_classes=3, ignored_classes=[1], cared_classes=[0])

    with pytest.raises(AssertionError):
        F1Score(num_classes=3, ignored_classes=1)

    with pytest.raises(AssertionError):
        F1Score(num_classes=2, mode=1)

    with pytest.raises(AssertionError):
        F1Score(num_classes=1, mode='1')


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([torch.LongTensor([0, 1, 2])], [torch.LongTensor([0, 1, 4])]),
        ([torch.LongTensor([0, 1]),
          torch.LongTensor([2])],
         [torch.LongTensor([0, 1]),
          torch.LongTensor([4])]),
    ],
)
def test_macro_metric_torch(predictions, labels):
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=5, mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.4)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)


@pytest.mark.skipif(
    flow is None or LooseVersion(flow.__version__) < '0.8.1',
    reason='OneFlow > 0.8.0 is required!')
@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([[0, 1, 2]], [[0, 1, 4]]),
        ([[0, 1], [2]], [[0, 1], [4]]),
    ],
)
def test_macro_metric_oneflow(predictions, labels):
    assertions = unittest.TestCase('__init__')
    predictions = list(
        flow.LongTensor(prediction) for prediction in predictions)
    labels = list(flow.LongTensor(label) for label in labels)
    f1 = F1Score(num_classes=5, mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.4)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)


@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([np.array([0, 1, 2])], [np.array([0, 1, 4])]),
        ([np.array([0, 1]), np.array([2])], [np.array([0, 1]),
                                             np.array([4])]),
    ])
def test_macro_metric_np(predictions, labels):
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=5, mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.4)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='macro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['macro_f1'], 0.25)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([torch.LongTensor([0, 1, 0, 1,
                            2])], [torch.LongTensor([0, 1, 2, 2, 0])]),
        ([torch.LongTensor([0, 1, 0]),
          torch.LongTensor([2, 2])],
         [torch.LongTensor([0, 1, 2]),
          torch.LongTensor([0, 1])]),
    ])
def test_micro_metric_torch(predictions, labels):
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=3, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)


@pytest.mark.skipif(
    flow is None or LooseVersion(flow.__version__) < '0.8.1',
    reason='OneFlow >= 0.8.1 is required!')
@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([[0, 1, 0, 1, 2]], [[0, 1, 2, 2, 0]]),
        ([[0, 1, 0], [2, 2]], [[0, 1, 2], [0, 1]]),
    ])
def test_micro_metric_oneflow(predictions, labels):
    assertions = unittest.TestCase('__init__')
    predictions = list(
        flow.LongTensor(prediction) for prediction in predictions)
    labels = list(flow.LongTensor(label) for label in labels)
    f1 = F1Score(num_classes=3, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)


@pytest.mark.parametrize(
    argnames=['predictions', 'labels'],
    argvalues=[
        ([np.array([0, 1, 0, 1, 2])], [np.array([0, 1, 2, 2, 0])]),
        ([np.array([0, 1, 0]),
          np.array([2, 2])], [np.array([0, 1, 2]),
                              np.array([0, 1])]),
    ])
def test_micro_metric_np(predictions, labels):
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=3, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)

    f1 = F1Score(num_classes=5, ignored_classes=[1], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)

    f1 = F1Score(num_classes=5, cared_classes=[0, 2, 3, 4], mode='micro')
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.285, delta=0.001)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_mode_torch():
    predictions = [torch.LongTensor([0, 1, 0, 1, 2])]
    labels = [torch.LongTensor([0, 1, 2, 2, 0])]
    mode = ['micro', 'macro']
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=3, mode=mode)
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)
    assertions.assertAlmostEqual(results['macro_f1'], 0.39, delta=0.01)


@pytest.mark.skipif(
    flow is None or LooseVersion(flow.__version__) < '0.8.1',
    reason='OneFlow >= 0.8.1 is required!')
def test_mode_oneflow():
    predictions = [flow.LongTensor([0, 1, 0, 1, 2])]
    labels = [flow.LongTensor([0, 1, 2, 2, 0])]
    mode = ['micro', 'macro']
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=3, mode=mode)
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)
    assertions.assertAlmostEqual(results['macro_f1'], 0.39, delta=0.01)


def test_mode_np():
    predictions = [np.array([0, 1, 0, 1, 2])]
    labels = [np.array([0, 1, 2, 2, 0])]
    mode = ['micro', 'macro']
    assertions = unittest.TestCase('__init__')

    f1 = F1Score(num_classes=3, mode=mode)
    results = f1(predictions, labels)
    assert isinstance(results, dict)
    assertions.assertAlmostEqual(results['micro_f1'], 0.4, delta=0.01)
    assertions.assertAlmostEqual(results['macro_f1'], 0.39, delta=0.01)
