# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.metrics import BLEU


def test_bleu():
    predictions = [
        'the cat is on the mat',
        'There is a big tree near the park here',
        'The sun rises from the northeast with sunshine',
        'I was late for work today for the rainy',
        'My name is Barry',
    ]
    references = [['a cat is on the mat'],
                  ['A big tree is growing near the park here'],
                  ['A fierce sun rises in the northeast with sunshine'],
                  ['I went to work too late today for the rainy'],
                  ['I am Barry']]

    bleu = BLEU()
    for i in range(len(predictions)):
        bleu.add([predictions[i]], [references[i]])
    bleu_results = bleu.compute()
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.4006741)

    bleu = BLEU(smooth=True)
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.4250477)

    predictions = ['猫坐在垫子上', '公园旁边有棵树']
    references = [['猫在那边的垫子'], ['一棵树长在公园旁边']]
    metric = BLEU()
    metric.add(predictions, references)
    bleu_results = metric.compute()
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.2576968)


@pytest.mark.parametrize('n_gram', [1, 2, 3, 4])
def test_bleu_ngram(n_gram):
    predictions = [
        'the cat is on the mat',
        'There is a big tree near the park here',
        'The sun rises from the northeast with sunshine',
        'I was late for work today for the rainy',
        'My name is Barry',
    ]
    references = [['a cat is on the mat'],
                  ['A big tree is growing near the park here'],
                  ['A fierce sun rises in the northeast with sunshine'],
                  ['I went to work too late today for the rainy'],
                  ['I am Barry']]

    bleu = BLEU(n_gram=n_gram)
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
