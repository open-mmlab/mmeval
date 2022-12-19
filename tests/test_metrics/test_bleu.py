# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from collections import Counter

from mmeval.metrics import BLEU
from mmeval.metrics.bleu import get_n_gram


@pytest.mark.parametrize('n_gram', [2, 4])
def test_get_n_gram(n_gram):
    token = ['a', 'cat', 'is', 'on', 'the', 'mat']
    result = get_n_gram(token, n_gram)
    if n_gram == 2:
        counter = Counter({
            ('a', ): 1,
            ('cat', ): 1,
            ('is', ): 1,
            ('on', ): 1,
            ('the', ): 1,
            ('mat', ): 1,
            ('a', 'cat'): 1,
            ('cat', 'is'): 1,
            ('is', 'on'): 1,
            ('on', 'the'): 1,
            ('the', 'mat'): 1
        })
    else:
        counter = Counter({
            ('a', ): 1,
            ('cat', ): 1,
            ('is', ): 1,
            ('on', ): 1,
            ('the', ): 1,
            ('mat', ): 1,
            ('a', 'cat'): 1,
            ('cat', 'is'): 1,
            ('is', 'on'): 1,
            ('on', 'the'): 1,
            ('the', 'mat'): 1,
            ('a', 'cat', 'is'): 1,
            ('cat', 'is', 'on'): 1,
            ('is', 'on', 'the'): 1,
            ('on', 'the', 'mat'): 1,
            ('a', 'cat', 'is', 'on'): 1,
            ('cat', 'is', 'on', 'the'): 1,
            ('is', 'on', 'the', 'mat'): 1
        })
    assert result == counter


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
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.4006741601366701)

    bleu = BLEU(smooth=True)
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.42504770796962527)


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
