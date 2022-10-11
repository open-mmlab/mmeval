# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.classification.multi_label import MultiLabelMetric
from mmeval.core.base_metric import BaseMetric

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
    # test predictions with labels
    multi_label_metric = MultiLabelMetric(num_classes=2, **metric_kwargs)
    assert isinstance(multi_label_metric, BaseMetric)
    results = multi_label_metric(
        np.asarray([[0.1, 0.9], [0.5, 0.5]]), np.asarray([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 2}])
@pytest.mark.parametrize('preditions', [
    [1, 1],  # raw int indices
    [[1], [0, 1]],  # raw int indices
])
@pytest.mark.parametrize('labels', [
    [0, 1],  # raw int indices
    [[0, 1], [1]],  # raw int indices
])
def test_metric_interface_builtin(metric_kwargs, preditions, labels):
    """Test builtin inputs."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preditions, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 2, 'topk': 1}])
@pytest.mark.parametrize('preditions', [
    np.array([[0.1, 0.9], [0.5, 0.6]]),  # scores in a ndarray
    [np.array([0.1, 0.9]), np.array([0.5, 0.6])],  # scores in Sequence
])
@pytest.mark.parametrize('labels', [
    np.array([[1, 0], [0, 1]]),  # one-hot indices in a ndarray
    [np.array([1, 0]), np.array([0, 1])],  # one-hot indices in Sequence
    [np.array([0]), np.array([1])],  # raw indices in Sequence
])
def test_metric_interface_topk(metric_kwargs, preditions, labels):
    """Test scores inputs with topk."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preditions, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 2, 'topk': 1}])
@pytest.mark.parametrize('preditions', [
    torch.Tensor([[0.1, 0.9], [0.5, 0.6]]),  # scores in a Tensor
    [torch.Tensor([0.1, 0.9]), torch.Tensor([0.5, 0.6])],  # scores in Sequence
])
@pytest.mark.parametrize('labels', [
    torch.Tensor([[1, 0], [0, 1]]),  # one-hot indices in a Tensor
    # one-hot indices in Sequence
    [torch.Tensor([1, 0]), torch.Tensor([0, 1])],
    [torch.Tensor([0]), torch.Tensor([1])],  # raw indices in Sequence
])
@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch_topk(metric_kwargs, preditions, labels):
    """Test scores inputs with topk in torch."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preditions, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 2}])
@pytest.mark.parametrize('preditions', [
    np.array([[0.1, 0.9], [0.5, 0.6]]),  # scores in a ndarray
    [np.array([0.1, 0.9]), np.array([0.5, 0.6])],  # scores in Sequence
    np.array([[0, 1], [1, 1]]),  # one-hot indices in a ndarray
    [np.array([0, 1]), np.array([1, 1])],  # one-hot indices in Sequence
    [np.array([1]), np.array([0, 1])],  # raw indices in Sequence
])
@pytest.mark.parametrize('labels', [
    np.array([[1, 0], [0, 1]]),  # one-hot indices in a ndarray
    [np.array([1, 0]), np.array([0, 1])],  # one-hot indices in Sequence
    [np.array([0]), np.array([1])],  # raw indices in Sequence
])
def test_metric_interface(metric_kwargs, preditions, labels):
    """Test all kinds of inputs."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preditions, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize('metric_kwargs', [{'num_classes': 2}])
@pytest.mark.parametrize('preditions', [
    torch.Tensor([[0.1, 0.9], [0.5, 0.6]]),  # scores in a Tensor
    # scores in Sequence
    [torch.Tensor([0.1, 0.9]), torch.Tensor([0.5, 0.6])],
    torch.Tensor([[0, 1], [1, 1]]),  # one-hot indices in a Tensor
    # one-hot indices in Sequence
    [torch.Tensor([0, 1]), torch.Tensor([1, 1])],
    [torch.Tensor([1]), torch.Tensor([0, 1])],  # raw indices in Sequence
])
@pytest.mark.parametrize('labels', [
    torch.Tensor([[1, 0], [0, 1]]),  # one-hot indices in a Tensor
    # one-hot indices in Sequence
    [torch.Tensor([1, 0]), torch.Tensor([0, 1])],
    [torch.Tensor([0]), torch.Tensor([1])],  # raw indices in Sequence
])
@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch(metric_kwargs, preditions, labels):
    """Test all kinds of inputs in torch."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    results = multi_label_metric(preditions, labels)
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'preditions', 'labels', 'results'],
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
def test_metric_accurate(metric_kwargs, preditions, labels, results):
    """Test accurate."""
    multi_label_metric = MultiLabelMetric(**metric_kwargs)
    assert multi_label_metric(
        np.asarray(preditions), np.asarray(labels)) == results


def test_metric_accurate_is_rawindex():
    """Test ambiguous cases when num_classes=2."""
    multi_label_metric = MultiLabelMetric(num_classes=2, items=('precision', 'recall')) # noqa
    assert multi_label_metric.pred_is_rawindex is False
    assert multi_label_metric.label_is_rawindex is False
    assert multi_label_metric([[0, 1], [1, 0]], [[0, 1], [0, 1]]) == {'precision': 50.0, 'recall': 25.0} # noqa
    multi_label_metric.pred_is_rawindex = True
    assert multi_label_metric([[0, 1], [1, 0]], [[0, 1], [0, 1]]) == {'precision': 50.0, 'recall': 50.0} # noqa
    multi_label_metric.label_is_rawindex = True
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

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = multi_label_metric(predictions, labels)

    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    torch_acc_results = multi_label_metric(predictions, labels)

    assert np_acc_results.keys() == torch_acc_results.keys()
    for key in np_acc_results:
        np.testing.assert_allclose(np_acc_results[key], torch_acc_results[key])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
