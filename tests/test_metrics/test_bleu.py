# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import Bleu


def test_bleu():
    predictions = [
        'the cat is on the mat', 'There is a big tree near the park here',
        'The sun rises from the northeast with sunshine',
        'I was late for work today for the rainy'
    ]
    references = [['a cat is on the mat'],
                  ['A big tree is growing near the park here'],
                  ['A fierce sun rises in the northeast with sunshine'],
                  ['I went to work too late today for the rainy']]

    bleu = Bleu()
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.4181)

    bleu = Bleu(smooth=True)
    bleu_results = bleu(predictions, references)
    assert isinstance(bleu_results, dict)
    np.testing.assert_almost_equal(bleu_results['bleu'], 0.442571)
