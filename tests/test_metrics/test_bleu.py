# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.metrics import Bleu


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

    bleu = Bleu()
    for i in range(len(predictions)):
        bleu.add([predictions[i]], [references[i]])
    bleu_results = bleu.compute()
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.4006741601366701)

    bleu = Bleu(smooth=True)
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

    bleu = Bleu(n_gram=n_gram)
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
