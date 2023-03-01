# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.metrics import SacreBLEU


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'metric_results'],
    argvalues=[
        ({
            'lowercase': False,
            'tokenizer_fn': None
        }, {
            'bleu': 0.4181
        }),
        ({
            'lowercase': False,
            'tokenizer_fn': 'none'
        }, {
            'bleu': 0.4181
        }),
        ({
            'lowercase': False,
            'tokenizer_fn': '13a'
        }, {
            'bleu': 0.4181
        }),
        ({
            'lowercase': False,
            'tokenizer_fn': 'intl'
        }, {
            'bleu': 0.4181
        }),
        ({
            'lowercase': False,
            'tokenizer_fn': 'char'
        }, {
            'bleu': 0.727862
        }),
        ({
            'lowercase': True,
            'tokenizer_fn': None
        }, {
            'bleu': 0.4405397
        }),
        ({
            'lowercase': True,
            'tokenizer_fn': 'none'
        }, {
            'bleu': 0.4405397
        }),
        ({
            'lowercase': True,
            'tokenizer_fn': '13a'
        }, {
            'bleu': 0.4405397
        }),
        ({
            'lowercase': True,
            'tokenizer_fn': 'intl'
        }, {
            'bleu': 0.4405397
        }),
        ({
            'lowercase': True,
            'tokenizer_fn': 'char'
        }, {
            'bleu': 0.7366514
        }),
    ])
def test_tokenizer_fn(metric_kwargs, metric_results):

    predictions = [
        'the cat is on the mat', 'There is a big tree near the park here',
        'The sun rises from the northeast with sunshine',
        'I was late for work today for the rainy'
    ]
    references = [['a cat is on the mat'],
                  ['A big tree is growing near the park here'],
                  ['A fierce sun rises in the northeast with sunshine'],
                  ['I went to work too late today for the rainy']]
    metric = SacreBLEU(**metric_kwargs)
    metric.add(predictions, references)
    results = metric.compute()
    assert len(metric_results) == len(results)
    for k, v in metric_results.items():
        np.testing.assert_almost_equal(v, results[k])


@pytest.mark.parametrize('lowercase', [False, True])
def test_tokenizer_ch(lowercase):

    predictions = ['猫坐在垫子上', '公园旁边有棵树']
    references = [['猫在那边的垫子'], ['一棵树长在公园旁边']]

    metric = SacreBLEU(lowercase=lowercase, tokenizer_fn='zh')
    metric.add(predictions, references)
    result = metric.compute()
    np.testing.assert_almost_equal(result['bleu'], 0.257697, decimal=6)
