# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import EndPointError
from mmeval.utils import try_import

torch = try_import('torch')
flow = try_import('oneflow')


def test_metric_interface_numpy():
    epe = EndPointError()
    assert isinstance(epe, BaseMetric)

    results = epe(
        np.random.normal(size=(10, 10, 2)),
        np.random.normal(size=(10, 10, 2)))
    assert isinstance(results, dict)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch():
    epe = EndPointError()
    assert isinstance(epe, BaseMetric)

    results = epe(torch.randn(10, 10, 2), torch.randn(10, 10, 2))
    assert isinstance(results, dict)


@pytest.mark.skipif(flow is None, reason='OneFlow is not available!')
def test_metric_interface_oneflow():
    epe = EndPointError()
    assert isinstance(epe, BaseMetric)

    results = epe(flow.randn(10, 10, 2), flow.randn(10, 10, 2))
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['predictions', 'labels', 'valid_masks'],
    argvalues=[
        (
            np.array([[[10., 5.], [0.1, 3.]], [[3., 15.2], [2.4, 4.5]]]),
            np.array([[[10.1, 4.8], [10, 3.]], [[6., 10.2], [2.0, 4.1]]]),
            np.array([[1., 1.], [1., 0.3]])
        )
    ]
)
def test_metric_accurate(predictions, labels, valid_masks):
    epe_target = np.linalg.norm((predictions - labels), ord=2, axis=-1)
    epe_target = (epe_target[0].sum() + epe_target[1][0]) / 3

    epe_metric = EndPointError()
    epe_result = epe_metric(predictions, labels, valid_masks)
    np.testing.assert_allclose(epe_result['EPE'], epe_target)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metamorphic_numpy_pytorch():
    """Metamorphic testing for NumPy and PyTorch implementation."""

    epe = EndPointError()
    predictions = np.random.normal(size=(10, 10, 2))
    labels = np.random.normal(size=(10, 10, 2))
    np_results = epe(predictions, labels)

    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    torch_results = epe(predictions, labels)

    assert np_results.keys() == torch_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], torch_results[key], rtol=1e-06)


@pytest.mark.skipif(flow is None, reason='OneFlow is not available!')
def test_metamorphic_numpy_oneflow():
    """Metamorphic testing for NumPy and OneFlow implementation."""

    epe = EndPointError()
    predictions = np.random.normal(size=(10, 10, 2))
    labels = np.random.normal(size=(10, 10, 2))
    np_results = epe(predictions, labels)

    predictions = flow.from_numpy(predictions)
    labels = flow.from_numpy(labels)
    oneflow_results = epe(predictions, labels)

    assert np_results.keys() == oneflow_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], oneflow_results[key], rtol=1e-06)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
