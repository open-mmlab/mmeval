# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from collections import Counter

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
