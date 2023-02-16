# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest
from distutils.version import LooseVersion

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import Accuracy
from mmeval.utils import try_import

torch = try_import('torch')
tf = try_import('tensorflow')
paddle = try_import('paddle')
jnp = try_import('jax.numpy')
flow = try_import('oneflow')


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'topk': 1, 'thrs': 0.1},
        {'topk': (1, 2), 'thrs': (0.1, 0.2)},
    ]
)
def test_metric_interface_numpy(metric_kwargs):
    accuracy = Accuracy(**metric_kwargs)
    assert isinstance(accuracy, BaseMetric)
    assert isinstance(accuracy.topk, tuple)
    assert isinstance(accuracy.thrs, tuple)

    results = accuracy(np.asarray([1, 2, 3]), np.asarray([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        np.asarray([[0.1, 0.9], [0.5, 0.5]]), np.asarray([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch():
    accuracy = Accuracy()
    results = accuracy(torch.Tensor([1, 2, 3]), torch.Tensor([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        torch.Tensor([[0.1, 0.9], [0.5, 0.5]]), torch.Tensor([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
def test_metric_interface_oneflow():
    accuracy = Accuracy()
    results = accuracy(flow.Tensor([1, 2, 3]), flow.Tensor([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        flow.Tensor([[0.1, 0.9], [0.5, 0.5]]), flow.Tensor([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
def test_metric_interface_tf():
    accuracy = Accuracy()
    results = accuracy(
        tf.convert_to_tensor([1, 2, 3]), tf.convert_to_tensor([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        tf.convert_to_tensor([[0.1, 0.9], [0.5, 0.5]]),
        tf.convert_to_tensor([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
def test_metric_interface_paddle():
    accuracy = Accuracy()
    results = accuracy(
        paddle.to_tensor([1, 2, 3]), paddle.to_tensor([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        paddle.to_tensor([[0.1, 0.9], [0.5, 0.5]]),
        paddle.to_tensor([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.skipif(jnp is None, reason='JAX is not available!')
def test_metric_interface_jnp():
    accuracy = Accuracy()
    results = accuracy(
        jnp.asarray([1, 2, 3]), jnp.asarray([3, 2, 1]))
    assert isinstance(results, dict)
    results = accuracy(
        jnp.asarray([[0.1, 0.9], [0.5, 0.5]]),
        jnp.asarray([0, 1]))
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'predictions', 'labels', 'results'],
    argvalues=[
        ({}, [0, 2, 1, 3], [0, 1, 2, 3], {'top1': 0.5}),
        (
            {'topk': (1, 2, 3)},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'top1': 0.5, 'top2': 0.75, 'top3': 1.}
        ),
        (
            {'topk': 2, 'thrs': (0.1, 0.5)},
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.3, 0.4, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.9]
            ],
            [0, 1, 2, 3],
            {'top2_thr-0.10': 0.75, 'top2_thr-0.50': 0.5}
        )
    ]
)
def test_metric_accurate(metric_kwargs, predictions, labels, results):
    accuracy = Accuracy(**metric_kwargs)
    assert accuracy(np.asarray(predictions), np.asarray(labels)) == results


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({'topk': 1, 'thrs': 0.1}, 1000, 100),
        ({'topk': (1, 2, 3), 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'topk': (1, 2, 3), 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_pytorch(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and PyTorch implementation."""
    accuracy = Accuracy(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = accuracy(predictions, labels)

    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    torch_acc_results = accuracy(predictions, labels)

    assert np_acc_results.keys() == torch_acc_results.keys()

    for key in np_acc_results:
        np.testing.assert_allclose(np_acc_results[key], torch_acc_results[key])


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({'topk': 1, 'thrs': 0.1}, 1000, 100),
        ({'topk': (1, 2, 3), 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'topk': (1, 2, 3), 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_oneflow(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and OneFlow implementation."""
    accuracy = Accuracy(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = accuracy(predictions, labels)

    predictions = flow.from_numpy(predictions)
    labels = flow.from_numpy(labels)
    flow_acc_results = accuracy(predictions, labels)

    assert np_acc_results.keys() == flow_acc_results.keys()

    for key in np_acc_results:
        np.testing.assert_allclose(np_acc_results[key], flow_acc_results[key])


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({'topk': 1, 'thrs': 0.1}, 1000, 100),
        ({'topk': (1, 2, 3), 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'topk': (1, 2, 3), 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_tf(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and TensorFlow implementation."""
    accuracy = Accuracy(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = accuracy(predictions, labels)

    predictions = tf.convert_to_tensor(predictions)
    labels = tf.convert_to_tensor(labels)
    tf_acc_results = accuracy(predictions, labels)

    assert np_acc_results.keys() == tf_acc_results.keys()

    for key in np_acc_results:
        np.testing.assert_allclose(np_acc_results[key], tf_acc_results[key])


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({'topk': 1, 'thrs': 0.1}, 1000, 100),
        ({'topk': (1, 2, 3), 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'topk': (1, 2, 3), 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_paddle(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and Paddle implementation."""
    accuracy = Accuracy(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = accuracy(predictions, labels)

    predictions = paddle.to_tensor(predictions)
    labels = paddle.to_tensor(labels)
    paddle_acc_results = accuracy(predictions, labels)

    assert np_acc_results.keys() == paddle_acc_results.keys()

    for key in np_acc_results:
        np.testing.assert_allclose(
            np_acc_results[key], paddle_acc_results[key])


@pytest.mark.skipif(jnp is None, reason='JAX is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'classes_num', 'length'),
    argvalues=[
        ({}, 100, 100),
        ({'topk': 1, 'thrs': 0.1}, 1000, 100),
        ({'topk': (1, 2, 3), 'thrs': (0.1, 0.2)}, 1000, 10000),
        ({'topk': (1, 2, 3), 'thrs': (0.1, None)}, 999, 10002)
    ]
)
def test_metamorphic_numpy_jax(metric_kwargs, classes_num, length):
    """Metamorphic testing for NumPy and JAX implementation."""
    accuracy = Accuracy(**metric_kwargs)

    predictions = np.random.rand(length, classes_num)
    labels = np.random.randint(0, classes_num, length)

    np_acc_results = accuracy(predictions, labels)

    predictions = jnp.asarray(predictions)
    labels = jnp.asarray(labels)
    jnp_acc_results = accuracy(predictions, labels)

    assert np_acc_results.keys() == jnp_acc_results.keys()

    for key in np_acc_results:
        np.testing.assert_allclose(
            np_acc_results[key], jnp_acc_results[key])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
