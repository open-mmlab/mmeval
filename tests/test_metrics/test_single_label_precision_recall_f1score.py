# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest
from distutils.version import LooseVersion

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import SingleLabelPrecisionRecallF1score
from mmeval.utils import try_import

torch = try_import('torch')
flow = try_import('oneflow')


def test_metric_init_assertion():
    with pytest.raises(AssertionError,
                       match='Invalid `average` argument'):
        SingleLabelPrecisionRecallF1score(average='mean')
    with pytest.raises(AssertionError,
                       match='The metric map is not supported'):
        SingleLabelPrecisionRecallF1score(items=('map',))


def test_metric_assertion():
    single_label_metric = SingleLabelPrecisionRecallF1score()
    with pytest.raises(AssertionError,
                       match='Please specify `num_classes`'):
        single_label_metric(
            np.asarray([1, 2, 3]), np.asarray([3, 2, 1]))

    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=2)
    with pytest.raises(AssertionError,
                       match='Number of classes does not match'):
        single_label_metric(
            np.asarray([[0.1, 0.9, 0], [0.5, 0.5, 0]]), np.asarray([0, 1]))


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_torch_assertion():
    single_label_metric = SingleLabelPrecisionRecallF1score()
    with pytest.raises(AssertionError, match='Please specify `num_classes`'):
        single_label_metric(
            torch.Tensor([1, 2, 3]), torch.Tensor([3, 2, 1]))

    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=2)
    with pytest.raises(AssertionError,
                       match='Number of classes does not match'):
        single_label_metric(
            torch.Tensor([[0.1, 0.9, 0], [0.5, 0.5, 0]]), torch.Tensor([0, 1]))


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
def test_metric_oneflow_assertion():
    single_label_metric = SingleLabelPrecisionRecallF1score()
    with pytest.raises(AssertionError, match='Please specify `num_classes`'):
        single_label_metric(
            flow.Tensor([1, 2, 3]), flow.Tensor([3, 2, 1]))

    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=2)
    with pytest.raises(AssertionError,
                       match='Number of classes does not match'):
        single_label_metric(
            flow.Tensor([[0.1, 0.9, 0], [0.5, 0.5, 0]]), flow.Tensor([0, 1]))


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'thrs': 0.1},
        {
            'thrs': (0.1, 0.2),
            'items': ('precision', 'recall', 'f1-score', 'support')
        },
        {'average': None},
        {'thrs': None, 'average': 'micro'},
    ]
)
def test_metric_interface(metric_kwargs):
    # test predictions with labels
    single_label_metric = SingleLabelPrecisionRecallF1score(**metric_kwargs)
    assert isinstance(single_label_metric, BaseMetric)
    assert isinstance(single_label_metric.thrs, tuple)
    results = single_label_metric(
        np.asarray([[0.1, 0.9], [0.5, 0.5]]), np.asarray([0, 1]))

    # test predictions with pred_scores
    single_label_metric = SingleLabelPrecisionRecallF1score(
        **metric_kwargs, num_classes=4)
    assert isinstance(single_label_metric, BaseMetric)
    assert isinstance(single_label_metric.thrs, tuple)
    results = single_label_metric(
        np.asarray([1, 2, 3]), np.asarray([3, 2, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_input_torch():
    # test predictions with labels
    single_label_metric = SingleLabelPrecisionRecallF1score()
    results = single_label_metric(
        torch.Tensor([[0.1, 0.9], [0.5, 0.5]]), torch.Tensor([0, 1]))
    assert isinstance(results, dict)

    # test predictions with pred_scores
    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=4)
    results = single_label_metric(
        torch.Tensor([1, 2, 3]), torch.Tensor([3, 2, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
def test_metric_input_oneflow():
    # test predictions with labels
    single_label_metric = SingleLabelPrecisionRecallF1score()
    results = single_label_metric(
        flow.Tensor([[0.1, 0.9], [0.5, 0.5]]), flow.Tensor([0, 1]))
    assert isinstance(results, dict)

    # test predictions with pred_scores
    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=4)
    results = single_label_metric(
        flow.Tensor([1, 2, 3]), flow.Tensor([3, 2, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_input_builtin():
    # test predictions with labels
    single_label_metric = SingleLabelPrecisionRecallF1score()
    results = single_label_metric(
        [[0.1, 0.9], [0.5, 0.5]], [0, 1])
    assert isinstance(results, dict)

    # test predictions with pred_scores
    single_label_metric = SingleLabelPrecisionRecallF1score(num_classes=4)
    results = single_label_metric(
        [1, 2, 3], [3, 2, 1])
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'predictions', 'labels', 'results'],
    argvalues=[
        ({'num_classes': 4}, [0, 2, 1, 3], [0, 1, 2, 3], {'precision': 50.0, 'recall': 50.0, 'f1-score': 50.0}), # noqa
        (
            {'average': 'micro'},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'precision_micro': 50.0, 'recall_micro': 50.0, 'f1-score_micro': 50.0} # noqa
        ),
        (
            {'average': None},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'f1-score_classwise': [100.0, 0.0, 0.0, 100.0], 'precision_classwise': [100.0, 0.0, 0.0, 100.0], 'recall_classwise': [100.0, 0.0, 0.0, 100.0]} # noqa
        )
    ]
)
def test_metric_accurate(metric_kwargs, predictions, labels, results):
    single_label_metric = SingleLabelPrecisionRecallF1score(**metric_kwargs)
    assert single_label_metric(
        np.asarray(predictions), np.asarray(labels)) == results


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({'num_classes': 100}, 100, 100),
        ({'num_classes': 1000, 'thrs': 0.1}, 1000, 100),
        ({'num_classes': 1000, 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'num_classes': 999, 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_pytorch(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and PyTorch implementation."""
    single_label_metric = SingleLabelPrecisionRecallF1score(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = single_label_metric(predictions, labels)

    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    torch_acc_results = single_label_metric(predictions, labels)

    assert np_acc_results.keys() == torch_acc_results.keys()
    print(np_acc_results, torch_acc_results)
    # torch defaults to float32 whereas numpy uses double
    for key in np_acc_results:
        np.testing.assert_allclose(
            np_acc_results[key], torch_acc_results[key], rtol=1e-5)


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({'num_classes': 100}, 100, 100),
        ({'num_classes': 1000, 'thrs': 0.1}, 1000, 100),
        ({'num_classes': 1000, 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'num_classes': 999, 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_oneflow(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and OneFlow implementation."""
    single_label_metric = SingleLabelPrecisionRecallF1score(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = single_label_metric(predictions, labels)

    predictions = flow.from_numpy(predictions)
    labels = flow.from_numpy(labels)
    oneflow_acc_results = single_label_metric(predictions, labels)

    assert np_acc_results.keys() == oneflow_acc_results.keys()
    print(np_acc_results, oneflow_acc_results)
    # oneflow defaults to float32 whereas numpy uses double
    for key in np_acc_results:
        np.testing.assert_allclose(
            np_acc_results[key], oneflow_acc_results[key], rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
