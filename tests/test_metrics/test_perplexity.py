import numpy as np
import pytest

from mmeval.metrics import Perplexity
from mmeval.utils import try_import

torch = try_import('torch')
tf = try_import('tensorflow')
paddle = try_import('paddle')
flow = try_import('oneflow')


@pytest.mark.skipif(torch is None, reason='Pytorch is not available!')
def test_input_shape():
    metric = Perplexity()
    dataset = {
        'pred': torch.rand(2, 8, generator=torch.manual_seed(22)),
        'target': torch.randint(5, (2, 8), generator=torch.manual_seed(22))
    }
    with pytest.raises(ValueError):
        metric.add(dataset['pred'], dataset['target'])
    dataset = {
        'pred': torch.rand(2, 8, 5, generator=torch.manual_seed(22)),
        'target': torch.randint(5, (2, 8, 4), generator=torch.manual_seed(22))
    }
    with pytest.raises(ValueError):
        metric.add(dataset['pred'], dataset['target'])
    dataset = {
        'pred': torch.rand(2, 10, 5, generator=torch.manual_seed(22)),
        'target': torch.randint(5, (2, 8), generator=torch.manual_seed(22))
    }
    with pytest.raises(ValueError):
        metric.add(dataset['pred'], dataset['target'])


@pytest.mark.skipif(torch is None, reason='Pytorch is not available!')
def test_perplexity_torch():
    np.random.seed(0)
    preds = np.random.rand(4, 2, 8, 5)
    targets = np.random.randint(low=0, high=5, size=(4, 2, 8))
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    metric = Perplexity()
    for i in range(4):
        pred = preds[i]
        target = targets[i]
        metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 5.235586)

    np.random.seed(0)
    pred = np.random.rand(1, 4, 4)
    target = np.random.randint(low=0, high=4, size=(1, 4))
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    metric = Perplexity(ignore_labels=[-101, -100, -101])
    target[:, 2] = -101
    target[:, 3] = -100
    metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 4.544293)


@pytest.mark.skipif(flow is None, reason='Oneflow is not available!')
def test_perplexity_oneflow():
    np.random.seed(0)
    preds = np.random.rand(4, 2, 8, 5)
    targets = np.random.randint(low=0, high=5, size=(4, 2, 8))
    preds = flow.as_tensor(preds)
    targets = flow.as_tensor(targets)
    metric = Perplexity()
    for i in range(4):
        pred = preds[i]
        target = targets[i]
        metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 5.235586)

    np.random.seed(0)
    pred = np.random.rand(1, 4, 4)
    target = np.random.randint(low=0, high=4, size=(1, 4))
    pred = flow.as_tensor(pred)
    target = flow.as_tensor(target)
    metric = Perplexity(ignore_labels=[-101, -100, -101])
    target[:, 2] = -101
    target[:, 3] = -100
    metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 4.544293)


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
def test_perplexity_tensorflow():
    np.random.seed(0)
    preds = np.random.rand(4, 2, 8, 5)
    targets = np.random.randint(low=0, high=5, size=(4, 2, 8))
    preds = tf.convert_to_tensor(preds)
    targets = tf.convert_to_tensor(targets)
    metric = Perplexity()
    for i in range(4):
        pred = preds[i]
        target = targets[i]
        metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 5.235586)

    np.random.seed(0)
    pred = np.random.rand(1, 4, 4)
    target = np.random.randint(low=0, high=4, size=(1, 4))
    metric = Perplexity(ignore_labels=[-101, -100, -101])
    target[:, 2] = -101
    target[:, 3] = -100
    pred = tf.convert_to_tensor(pred)
    target = tf.convert_to_tensor(target)
    metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 4.544293)


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
def test_perplexity_paddle():
    np.random.seed(0)
    preds = np.random.rand(4, 2, 8, 5)
    targets = np.random.randint(low=0, high=5, size=(4, 2, 8))
    preds = paddle.to_tensor(preds)
    targets = paddle.to_tensor(targets)
    metric = Perplexity()
    for i in range(4):
        pred = preds[i]
        target = targets[i]
        metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 5.235586)

    np.random.seed(0)
    pred = np.random.rand(1, 4, 4)
    target = np.random.randint(low=0, high=4, size=(1, 4))
    pred = paddle.to_tensor(pred)
    target = paddle.to_tensor(target)
    metric = Perplexity(ignore_labels=[-101, -100, -101])
    target[:, 2] = -101
    target[:, 3] = -100
    metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 4.544293)


def test_perplexity_numpy():
    np.random.seed(0)
    preds = np.random.rand(4, 2, 8, 5)
    targets = np.random.randint(low=0, high=5, size=(4, 2, 8))
    metric = Perplexity()
    for i in range(4):
        pred = preds[i]
        target = targets[i]
        metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 5.235586)

    np.random.seed(0)
    pred = np.random.rand(1, 4, 4)
    target = np.random.randint(low=0, high=4, size=(1, 4))
    metric = Perplexity(ignore_labels=[-101, -100, -101])
    target[:, 2] = -101
    target[:, 3] = -100
    metric.add(pred, target)
    my_result = metric.compute()
    result = my_result['perplexity']
    np.testing.assert_almost_equal(result, 4.544293)
