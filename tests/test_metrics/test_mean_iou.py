# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest
from distutils.version import LooseVersion

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import MeanIoU
from mmeval.utils import try_import

torch = try_import('torch')
jax = try_import('jax')
jnp = try_import('jax.numpy')
paddle = try_import('paddle')
tf = try_import('tensorflow')
flow = try_import('oneflow')


def test_metric_interface_numpy():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        np.random.randint(0, 4, size=(2, 10, 10)),
        np.random.randint(0, 4, size=(2, 10, 10))
    )
    assert isinstance(results, dict)


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_metric_interface_torch():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        torch.randint(0, 4, size=(2, 10, 10)),
        torch.randint(0, 4, size=(2, 10, 10))
    )
    assert isinstance(results, dict)


@pytest.mark.skipif(jnp is None, reason='JAX is not available!')
def test_metric_interface_jnp():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        jax.random.randint(jax.random.PRNGKey(0), (2, 10, 10), 0, 4),
        jax.random.randint(jax.random.PRNGKey(0), (2, 10, 10), 0, 4)
    )
    assert isinstance(results, dict)


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
def test_metric_interface_oneflow():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        flow.randint(0, 4, size=(2, 10, 10)),
        flow.randint(0, 4, size=(2, 10, 10))
    )
    assert isinstance(results, dict)


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
def test_metric_interface_paddle():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        paddle.randint(0, 4, shape=(2, 10, 10)),
        paddle.randint(0, 4, shape=(2, 10, 10))
    )
    assert isinstance(results, dict)


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
def test_metric_interface_tf():
    miou = MeanIoU(num_classes=4)
    assert isinstance(miou, BaseMetric)

    results = miou(
        tf.random.uniform((2, 10, 10), minval=0, maxval=4, dtype=tf.int32),
        tf.random.uniform((2, 10, 10), minval=0, maxval=4, dtype=tf.int32)
    )
    assert isinstance(results, dict)


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'predictions', 'labels', 'results'],
    argvalues=[
        (
            # for this test case argvalues
            # confusion matrix:
            #  2, 0, 0, 0
            #  0, 1, 1, 0
            #  0, 1, 1, 0
            #  0, 0, 0, 2
            #
            #  IoU = [2, 1, 1, 2] / [2, 3, 3, 2]
            #      = [1, 1/3, 1/3, 1]
            #  mIoU ~= 0.666666667
            {'num_classes': 4},
            [[[0, 2, 1, 3], [0, 2, 1, 3]], ],
            [[[0, 1, 2, 3], [0, 2, 1, 3]], ],
            {
                'mIoU': 0.666666667,
                'aAcc': 0.75,
                'mAcc': 0.75,
                'mDice': 0.75,
                'mPrecision': 0.75,
                'mRecall': 0.75,
                'mFscore': 0.75,
                'kappa': 0.666666667,
            }
        ),
    ]
)
def test_metric_accurate(metric_kwargs, predictions, labels, results):
    miou = MeanIoU(**metric_kwargs)
    metric_results = miou(np.asarray(predictions), np.asarray(labels))
    assert metric_results.keys() == results.keys()

    for key in metric_results:
        np.testing.assert_allclose(metric_results[key], results[key])


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'length'),
    argvalues=[
        ({'num_classes': 10}, 100),
        ({'num_classes': 100}, 1000),
        ({'num_classes': 222}, 500)
    ]
)
def test_metamorphic_numpy_pytorch(metric_kwargs, length):
    """Metamorphic testing for NumPy and PyTorch implementation."""
    miou = MeanIoU(**metric_kwargs)
    num_classes = metric_kwargs.get('num_classes')

    predictions = np.random.randint(0, num_classes, size=(length, 224, 224))
    labels = np.random.randint(0, num_classes, size=(length, 224, 224))

    np_results = miou(predictions, labels)

    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    torch_results = miou(predictions, labels)

    assert np_results.keys() == torch_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], torch_results[key], rtol=1e-06)


@pytest.mark.skipif(jnp is None, reason='JAX is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'length'),
    argvalues=[
        ({'num_classes': 10}, 100),
        ({'num_classes': 100}, 1000),
        ({'num_classes': 222}, 500)
    ]
)
def test_metamorphic_numpy_jnp(metric_kwargs, length):
    """Metamorphic testing for NumPy and JAX implementation."""
    miou = MeanIoU(**metric_kwargs)
    num_classes = metric_kwargs.get('num_classes')

    predictions = np.random.randint(0, num_classes, size=(length, 224, 224))
    labels = np.random.randint(0, num_classes, size=(length, 224, 224))

    np_results = miou(predictions, labels)

    predictions = jnp.asarray(predictions)
    labels = jnp.asarray(labels)
    jnp_results = miou(predictions, labels)

    assert np_results.keys() == jnp_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], jnp_results[key], rtol=1e-06)


@pytest.mark.skipif(flow is None or
                    LooseVersion(flow.__version__) < '0.8.1',
                    reason='OneFlow >= 0.8.1 is required!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'length'),
    argvalues=[
        ({'num_classes': 10}, 100),
        ({'num_classes': 100}, 1000),
        ({'num_classes': 222}, 500)
    ]
)
def test_metamorphic_numpy_oneflow(metric_kwargs, length):
    """Metamorphic testing for NumPy and OneFlow implementation."""
    miou = MeanIoU(**metric_kwargs)
    num_classes = metric_kwargs.get('num_classes')

    predictions = np.random.randint(0, num_classes, size=(length, 224, 224))
    labels = np.random.randint(0, num_classes, size=(length, 224, 224))

    np_results = miou(predictions, labels)

    predictions = flow.from_numpy(predictions)
    labels = flow.from_numpy(labels)
    oneflow_results = miou(predictions, labels)

    assert np_results.keys() == oneflow_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], oneflow_results[key], rtol=1e-06)


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'length'),
    argvalues=[
        ({'num_classes': 10}, 100),
        ({'num_classes': 100}, 1000),
        ({'num_classes': 222}, 500)
    ]
)
def test_metamorphic_numpy_paddle(metric_kwargs, length):
    """Metamorphic testing for NumPy and PaddlePaddle implementation."""
    miou = MeanIoU(**metric_kwargs)
    num_classes = metric_kwargs.get('num_classes')

    predictions = np.random.randint(0, num_classes, size=(length, 224, 224))
    labels = np.random.randint(0, num_classes, size=(length, 224, 224))

    np_results = miou(predictions, labels)

    predictions = paddle.to_tensor(predictions)
    labels = paddle.to_tensor(labels)
    paddle_results = miou(predictions, labels)

    assert np_results.keys() == paddle_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], paddle_results[key], rtol=1e-06)


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
@pytest.mark.parametrize(
    argnames=('metric_kwargs', 'length'),
    argvalues=[
        ({'num_classes': 10}, 100),
        ({'num_classes': 100}, 1000),
        ({'num_classes': 222}, 500)
    ]
)
def test_metamorphic_numpy_tf(metric_kwargs, length):
    """Metamorphic testing for NumPy and TensorFlow implementation."""
    miou = MeanIoU(**metric_kwargs)
    num_classes = metric_kwargs.get('num_classes')

    predictions = np.random.randint(0, num_classes, size=(length, 224, 224))
    labels = np.random.randint(0, num_classes, size=(length, 224, 224))

    np_results = miou(predictions, labels)

    predictions = tf.convert_to_tensor(predictions)
    labels = tf.convert_to_tensor(labels)
    tf_results = miou(predictions, labels)

    assert np_results.keys() == tf_results.keys()

    for key in np_results:
        np.testing.assert_allclose(
            np_results[key], tf_results[key], rtol=1e-06)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
