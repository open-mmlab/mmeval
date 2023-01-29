import numpy as np
import pytest

from mmeval.metrics import ROUGE


@pytest.mark.parametrize('dataset', [
    {
        'predictions': ['猫坐在垫子上', '公园旁边有棵树'],
        'references': [['猫在那边的垫子'], ['一棵树长在公园旁边']]
    },
])
def test_chinese(dataset):
    metric = ROUGE()
    metric.add(dataset['predictions'], dataset['references'])
    results = metric.compute()
    np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.3766233)
    np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.5576923)


@pytest.mark.parametrize('rouge_keys', [2, 'L', [2, 'L']])
def test_rouge_keys(rouge_keys):
    predictions = [
        'the cat is on the mats', 'There is a big trees near the park here',
        'The sun rises from the northeast with sunshine',
        'I was later for work today for the rainy'
    ]
    references = [[
        'a cat is on the mat', 'one cat is in the mat', 'cat is in mat',
        'a cat is on a blue mat'
    ], ['A big tree is growing near the parks here'],
                  ['A fierce sunning rises in the northeast with sunshine'],
                  ['I go to working too later today for the rainy']]

    metric = ROUGE(rouge_keys=rouge_keys)
    results = metric(predictions, references)
    if rouge_keys == 2:
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352)
    elif rouge_keys == 'L':
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105091)
    else:
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105091)


@pytest.mark.parametrize('accumulate', ['best', 'avg'])
def test_rouge(accumulate):
    predictions = [
        'the cat is on the mats', 'There is a big trees near the park here',
        'The sun rises from the northeast with sunshine',
        'I was later for work today for the rainy'
    ]
    references = [[
        'a cat is on the mat', 'one cat is in the mat', 'cat is in mat',
        'a cat is on a blue mat'
    ], ['A big tree is growing near the parks here'],
                  ['A fierce sunning rises in the northeast with sunshine'],
                  ['I go to working too later today for the rainy']]

    metric = ROUGE(accumulate=accumulate)
    results = metric(predictions, references)

    if accumulate == 'best':
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.6382868)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105090)
    else:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.5983830)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.5706052)


def test_rouge_lowercase():
    predictions = ['There is a big trees near the park here']
    references = [['A big tree is growing near the parks here']]

    metric = ROUGE(lowercase=False)
    results = metric(predictions, references)

    np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.5555555)
    np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.4444444)
