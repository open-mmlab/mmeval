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
@pytest.mark.parametrize(
    argnames=['preds', 'targets', 'metric_kwargs', 'metric_results'],
    argvalues=[
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, 0, 0]],
            {
                'ignore_labels': None
            },
            {
                'perplexity': 4.491682289784082
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -100]],
            {
                'ignore_labels': -100
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -101]],
            {
                'ignore_labels': [-100, -101]
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
    ])
def test_perplexity_torch(preds, targets, metric_kwargs, metric_results):
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    metric = Perplexity(**metric_kwargs)
    results = metric(preds, targets)

    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k])


@pytest.mark.skipif(flow is None, reason='Oneflow is not available!')
@pytest.mark.parametrize(
    argnames=['preds', 'targets', 'metric_kwargs', 'metric_results'],
    argvalues=[
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, 0, 0]],
            {
                'ignore_labels': None
            },
            {
                'perplexity': 4.491682289784082
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -100]],
            {
                'ignore_labels': -100
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -101]],
            {
                'ignore_labels': [-100, -101]
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
    ])
def test_perplexity_oneflow(preds, targets, metric_kwargs, metric_results):
    preds = flow.as_tensor(preds)
    targets = flow.as_tensor(targets)
    metric = Perplexity(**metric_kwargs)
    results = metric(preds, targets)

    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k])


@pytest.mark.skipif(tf is None, reason='TensorFlow is not available!')
@pytest.mark.parametrize(
    argnames=['preds', 'targets', 'metric_kwargs', 'metric_results'],
    argvalues=[
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, 0, 0]],
            {
                'ignore_labels': None
            },
            {
                'perplexity': 4.491682289784082
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -100]],
            {
                'ignore_labels': -100
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -101]],
            {
                'ignore_labels': [-100, -101]
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
    ])
def test_perplexity_tensorflow(preds, targets, metric_kwargs, metric_results):
    preds = tf.convert_to_tensor(preds)
    targets = tf.convert_to_tensor(targets)
    metric = Perplexity(**metric_kwargs)
    results = metric(preds, targets)

    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k], decimal=6)


@pytest.mark.skipif(paddle is None, reason='Paddle is not available!')
@pytest.mark.parametrize(
    argnames=['preds', 'targets', 'metric_kwargs', 'metric_results'],
    argvalues=[
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, 0, 0]],
            {
                'ignore_labels': None
            },
            {
                'perplexity': 4.491682289784082
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -100]],
            {
                'ignore_labels': -100
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -101]],
            {
                'ignore_labels': [-100, -101]
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
    ])
def test_perplexity_paddle(preds, targets, metric_kwargs, metric_results):
    preds = paddle.to_tensor(preds)
    targets = paddle.to_tensor(targets)
    metric = Perplexity(**metric_kwargs)
    results = metric(preds, targets)

    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k])


@pytest.mark.parametrize(
    argnames=['preds', 'targets', 'metric_kwargs', 'metric_results'],
    argvalues=[
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, 0, 0]],
            {
                'ignore_labels': None
            },
            {
                'perplexity': 4.491682289784082
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -100]],
            {
                'ignore_labels': -100
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
        (
            [[[0.27032791, 0.1314828, 0.05537432, 0.30159863],
              [0.26211815, 0.45614057, 0.68328134, 0.69562545],
              [0.28351885, 0.37992696, 0.18115096, 0.78854551],
              [0.05684808, 0.69699724, 0.7786954, 0.77740756]]],
            [[3, 2, -100, -101]],
            {
                'ignore_labels': [-100, -101]
            },
            {
                'perplexity': 3.529430098122886
            },
        ),
    ])
def test_perplexity_numpy(preds, targets, metric_kwargs, metric_results):
    preds = np.array(preds)
    targets = np.array(targets)
    metric = Perplexity(**metric_kwargs)
    results = metric(preds, targets)

    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k])
