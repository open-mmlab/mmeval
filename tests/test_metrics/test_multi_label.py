# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import MultiLabelMetric

try:
    import torch
except ImportError:
    torch = None


def test_metric_init_assertion():
    with pytest.raises(AssertionError,
                       match='Invalid `average` argument'):
        MultiLabelMetric(3, average='mean')
    with pytest.raises(AssertionError,
                       match='The metric map is not supported'):
        MultiLabelMetric(3, items=('map',))


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'thr': 0.1},
        {'topk': 2},
        {'thr': 0.1, 'topk': 2},
        {
            'thr': 0.2,
            'items': ('precision', 'recall', 'f1-score', 'support')
        },
        {'average': None},
        {'thr': None, 'average': 'micro'},
    ]
)
def test_metric_inputs(metric_kwargs):
    # test predictions with targets
    multi_label_metric = MultiLabelMetric(num_classes=3, **metric_kwargs)
    assert isinstance(multi_label_metric, BaseMetric)
    results = multi_label_metric(
        np.asarray([[0.1, 0.9, 0.8], [0.5, 0.5, 0.8]]), np.asarray([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 3}])
@pytest.mark.parametrize('preds', [
    [1, 1, 2],  # label-format predictions
    [[1], [0, 1], [2]],  # label-format predictions
])
@pytest.mark.parametrize('targets', [
    [0, 1, 2],  # label-format targets
    [[0, 1], [1], [2]],  # label-format targets
])
def test_metric_interface_builtin(metric_kwargs, preds, targets):
    """Test builtin inputs."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preds, targets)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 3, 'topk': 1}])
@pytest.mark.parametrize('preds', [
    np.array([[0.1, 0.9, 0.8], [0.5, 0.6, 0.8]]),  # scores in a ndarray
    # scores in Sequence
    [np.array([0.1, 0.9, 0.8]), np.array([0.5, 0.6, 0.8])],
])
@pytest.mark.parametrize('targets', [
    np.array([[1, 0, 0], [0, 1, 0]]),  # one-hot encodings in a ndarray
    # one-hot encodings in Sequence
    [np.array([1, 0, 0]), np.array([0, 1, 0])],
    [np.array([0]), np.array([1])],  # label-format in Sequence
])
def test_metric_interface_topk(metric_kwargs, preds, targets):
    """Test scores inputs with topk."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preds, targets)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 3, 'topk': 1}])
@pytest.mark.parametrize('preds', [
    torch.Tensor([[0.1, 0.9, 0.8], [0.5, 0.6, 0.8]]),  # scores in a Tensor
    # scores in Sequence
    [torch.Tensor([0.1, 0.9, 0.8]), torch.Tensor([0.5, 0.6, 0.8])],
])
@pytest.mark.parametrize('targets', [
    torch.Tensor([[1, 0, 0], [0, 1, 0]]),  # one-hot encodings in a Tensor
    # one-hot encodings in Sequence
    [torch.Tensor([1, 0, 0]), torch.Tensor([0, 1, 0])],
    [torch.Tensor([0]), torch.Tensor([1])],  # label-format in Sequence
])
@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch_topk(metric_kwargs, preds, targets):
    """Test scores inputs with topk in torch."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preds, targets)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 3}])
@pytest.mark.parametrize('preds', [
    np.array([[0.1, 0.9, 0.8], [0.5, 0.6, 0.8]]),  # scores in a ndarray
    # scores in Sequence
    [np.array([0.1, 0.9, 0.8]), np.array([0.5, 0.6, 0.8])],
    np.array([[0, 1, 0], [1, 1, 0]]),  # one-hot encodings in a ndarray
    # one-hot encodings in Sequence
    [np.array([0, 1, 0]), np.array([1, 1, 0])],
    [np.array([1]), np.array([0, 1])],  # label-format in Sequence
])
@pytest.mark.parametrize('targets', [
    np.array([[1, 0, 0], [0, 1, 0]]),  # one-hot encodings in a ndarray
    # one-hot encodings in Sequence
    [np.array([1, 0, 0]), np.array([0, 1, 0])],
    [np.array([0]), np.array([1])],  # label-format in Sequence
])
def test_metric_interface(metric_kwargs, preds, targets):
    """Test all kinds of inputs."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preds, targets)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 3}])
@pytest.mark.parametrize('preds', [
    torch.Tensor([[0.1, 0.9, 0.8], [0.5, 0.6, 0.8]]),  # scores in a Tensor
    # scores in Sequence
    [torch.Tensor([0.1, 0.9, 0.8]), torch.Tensor([0.5, 0.6, 0.8])],
    torch.Tensor([[0, 1, 0], [1, 1, 0]]),  # one-hot encodings in a Tensor
    # one-hot encodings in Sequence
    [torch.Tensor([0, 1, 0]), torch.Tensor([1, 1, 0])],
    [torch.Tensor([1]), torch.Tensor([0, 1])],  # label-format in Sequence
])
@pytest.mark.parametrize('targets', [
    torch.Tensor([[1, 0, 0], [0, 1, 0]]),  # one-hot encodings in a Tensor
    # one-hot encodings in Sequence
    [torch.Tensor([1, 0, 0]), torch.Tensor([0, 1, 0])],
    [torch.Tensor([0]), torch.Tensor([1])],  # label-format in Sequence
])
@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch(metric_kwargs, preds, targets):
    """Test all kinds of inputs in torch."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preds, targets)
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'preds', 'targets', 'results'],
    argvalues=[
        ({'num_classes': 4}, [0, 2, 1, 3], [0, 1, 2, 3], {'precision': 50.0, 'recall': 50.0, 'f1-score': 50.0}), # noqa
        (
            {'num_classes': 4, 'average': 'micro', 'thr': 0.25},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'precision_thr-0.25_micro': 50.0, 'recall_thr-0.25_micro': 75.0, 'f1-score_thr-0.25_micro': 60.0} # noqa
        ),
        (
            {'num_classes': 4, 'average': None, 'topk': 1},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'f1-score_top1_classwise': [100.0, 0.0, 0.0, 100.0], 'precision_top1_classwise': [100.0, 0.0, 0.0, 100.0], 'recall_top1_classwise': [100.0, 0.0, 0.0, 100.0]} # noqa
        )
    ]
)
def test_metric_accurate(metric_kwargs, preds, targets, results):
    """Test accurate."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    assert multi_label_metric(
        np.asarray(preds), np.asarray(targets)) == results


def test_metric_accurate_is_label():
    """Test ambiguous cases when num_classes=2."""
    multi_label_metric = MultiLabelMetric(num_classes=2, items=('precision', 'recall')) # noqa
    assert multi_label_metric.pred_is_label is False
    assert multi_label_metric.target_is_label is False
    assert multi_label_metric([[0, 1], [1, 0]], [[0, 1], [0, 1]]) == {'precision': 50.0, 'recall': 25.0} # noqa
    multi_label_metric.pred_is_label = True
    assert multi_label_metric([[0, 1], [1, 0]], [[0, 1], [0, 1]]) == {'precision': 50.0, 'recall': 50.0} # noqa
    multi_label_metric.target_is_label = True
    assert multi_label_metric([[0, 1], [1, 0]], [[0, 1], [0, 1]]) == {'precision': 100.0, 'recall': 100.0} # noqa


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({'num_classes': 100}, 100, 100),
        ({'num_classes': 1000, 'thr': 0.1}, 1000, 100),
        ({'num_classes': 1000, 'topk': 2}, 1000, 10000),
        ({'num_classes': 999, 'average': None}, 999, 10002)
    ]
)
def test_metamorphic_numpy_pytorch(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and PyTorch implementation."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)

    preds = np.random.rand(length, classes_num)
    targets = np.random.randint(0, classes_num, length)

    np_acc_results = multi_label_metric(preds, targets)

    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    torch_acc_results = multi_label_metric(preds, targets)

    assert np_acc_results.keys() == torch_acc_results.keys()
    for key in np_acc_results:
        # precision is different between numpy and torch
        np.testing.assert_allclose(
            np_acc_results[key], torch_acc_results[key], rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
