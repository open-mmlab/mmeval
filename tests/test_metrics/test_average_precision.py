# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import AveragePrecision
from mmeval.utils import try_import

torch = try_import('torch')
flow = try_import('oneflow')


def test_metric_init_assertion():
    with pytest.raises(AssertionError,
                       match='Invalid `average` argument'):
        AveragePrecision(average='micro')


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'average': None},
    ]
)
def test_metric_interface(metric_kwargs):
    average_precision = AveragePrecision(**metric_kwargs)
    assert isinstance(average_precision, BaseMetric)
    results = average_precision(
        np.asarray([[0.1, 0.9], [0.5, 0.5]]), np.asarray([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.parametrize('preds', [
    [[0.1, 0.9], [0.5, 0.6]],  # builtin prediction scores
])
@pytest.mark.parametrize('labels', [
    [[1, 0], [0, 1]],  # builtin one-hot encodings labels
    [[0, 1], [1]],  # builtin label-format labels
])
def test_metric_input_builtin(preds, labels):
    """Test builtin inputs."""
    average_precision = AveragePrecision()
    results = average_precision(preds, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('preds', [
    torch.tensor([[0.1, 0.9], [0.5, 0.6]]),  # scores in a ndarray
    [torch.tensor([0.1, 0.9]), torch.tensor([0.5, 0.6])],  # scores in Sequence
])
@pytest.mark.parametrize('labels', [
    torch.tensor([[1, 0], [0, 1]]),  # one-hot encodings labels in a ndarray
    # one-hot encodings labels in Sequence
    [torch.tensor([1, 0]), torch.tensor([0, 1])],
    [torch.tensor([0]), torch.tensor([1])],  # label-format labels in Sequence
])
@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_input_torch(preds, labels):
    """Test torch inputs."""
    average_precision = AveragePrecision()
    results = average_precision(preds, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('preds', [
    np.array([[0.1, 0.9], [0.5, 0.6]]),  # scores in a ndarray
    [[0.1, 0.9], [0.5, 0.6]],  # scores in Sequence
])
@pytest.mark.parametrize('labels', [
    np.array([[1, 0], [0, 1]]),  # one-hot encodings labels in a ndarray
    # one-hot encodings labels in Sequence
    [[1, 0], [0, 1]],
    [[0], [1]],  # label-format labels in Sequence
])
@pytest.mark.skipif(flow is None, reason='OneFlow is not available!')
def test_metric_input_flow(preds, labels):
    """Test oneflow inputs."""
    if isinstance(preds, np.ndarray):
        preds = flow.tensor(preds)
    else:
        preds = list(flow.tensor(pred) for pred in preds)
    if isinstance(labels, np.ndarray):
        labels = flow.tensor(labels)
    else:
        labels = list(flow.tensor(label) for label in labels)
    average_precision = AveragePrecision()
    results = average_precision(preds, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('preds', [
    np.array([[0.1, 0.9], [0.5, 0.6]]),  # scores in a ndarray
    [np.array([0.1, 0.9]), np.array([0.5, 0.6])],  # scores in Sequence
])
@pytest.mark.parametrize('labels', [
    np.array([[1, 0], [0, 1]]),  # one-hot encodings labels in a ndarray
    # one-hot encodings labels in Sequence
    [np.array([1, 0]), np.array([0, 1])],
    [np.array([0]), np.array([1])],  # label-format labels in Sequence
])
def test_metric_input_numpy(preds, labels):
    """Test numpy inputs."""
    average_precision = AveragePrecision()
    results = average_precision(preds, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'preds', 'labels', 'results'],
    argvalues=[
        (
            {},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3], {'mAP': 75.0}
        ),
        (
            {'average': None},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3], {'AP_classwise': [100.0, 50.0, 50.0, 100.0]}
        )
    ]
)
def test_metric_accurate(metric_kwargs, preds, labels, results):
    """Test accurate."""
    average_precision = AveragePrecision(**metric_kwargs)
    _results = average_precision(
        np.asarray(preds), np.asarray(labels))
    for (k1, v1), (k2, v2) in zip(_results.items(), results.items()):
        assert k1 == k2
        assert np.allclose(v1, v2)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({}, 1000, 10000),
        ({'average': None}, 999, 10002)
    ]
)
def test_metamorphic_numpy_pytorch(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and PyTorch implementation."""
    average_precision = AveragePrecision(**metric_kwargs)

    preds = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = average_precision(preds, labels)

    preds = torch.from_numpy(preds)
    labels = torch.from_numpy(labels)
    torch_acc_results = average_precision(preds, labels)

    assert np_acc_results.keys() == torch_acc_results.keys()
    for key in np_acc_results:
        # numpy use float64 however torch use float32
        np.testing.assert_allclose(
            np_acc_results[key], torch_acc_results[key], atol=1e-4)


@pytest.mark.skipif(flow is None, reason='OneFlow is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({}, 1000, 10000),
        ({'average': None}, 999, 10002)
    ]
)
def test_metamorphic_numpy_oneflow(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and OneFlow implementation."""
    average_precision = AveragePrecision(**metric_kwargs)

    preds = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = average_precision(preds, labels)

    preds = flow.from_numpy(preds)
    labels = flow.from_numpy(labels)
    oneflow_acc_results = average_precision(preds, labels)

    assert np_acc_results.keys() == oneflow_acc_results.keys()
    for key in np_acc_results:
        # numpy use float64 however oneflow use float32
        np.testing.assert_allclose(
            np_acc_results[key], oneflow_acc_results[key], atol=1e-4)
