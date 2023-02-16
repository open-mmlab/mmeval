# Copyright (c) OpenMMLab. All rights reserved.
# This class is modified from `torchmetrics
# <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/bleu.py>`_.
import numpy as np
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple, Union

from mmeval import BaseMetric
from mmeval.metrics.utils import get_n_gram, get_tokenizer, infer_language


def _get_brevity_penalty(pred_len: np.array,
                         references_len: np.array) -> np.array:
    """This function is used to calculate penalty factor.

    Args:
        pred_len (np.array): number of grams in the predicted sentence.
        references_len (np.array): number of grams in the references.

    Returns:
        np.array: penalty factor.
    """
    if pred_len > references_len:
        return np.array(1.)
    return np.array(np.exp(1 - references_len / pred_len))


class BLEU(BaseMetric):
    """Bilingual Evaluation Understudy metric.

    This metric proposed in `BLEU: a Method for Automatic Evaluation of Machine Translation
    <https://aclanthology.org/P02-1040.pdf>`_ is a tool for evaluating the quality of machine translation.
    The closer the translation is to human translation,
    the higher the score will be.

    Args:
        n_gram (int): The maximum number of words contained in a phrase
            when calculating word fragments. Defaults to 4.
        smooth (bool): Whether or not to apply to smooth. Defaults to False.
        ngram_weights (Sequence[float], optional): Weights used
            for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used. Defaults to None.
        tokenizer_fn (Union[Callable, str, None]): A user's own tokenizer function.
            Defaults to None.
            New in version 0.3.0.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> from mmeval import BLEU
        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here']  # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']]  # noqa: E501
        >>> bleu = BLEU()
        >>> bleu_results = bleu(predictions, references)
        {'bleu': 0.5226045319355426}

        >>> # Calculate BLEU with smooth:
        >>> from mmeval import BLEU
        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here']  # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']]  # noqa: E501
        >>> bleu = BLEU(smooth = True)
        >>> bleu_results = bleu(predictions, references)
        {'bleu': 0.566315716093867}
    """

    def __init__(self,
                 n_gram: int = 4,
                 smooth: bool = False,
                 ngram_weights: Optional[Sequence[float]] = None,
                 tokenizer_fn: Union[Callable, str, None] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.smooth = smooth
        if ngram_weights is not None and len(ngram_weights) != n_gram:
            raise ValueError(
                'The length of ngram_weights is not equal to `n_gram`: '
                f'{len(ngram_weights)} != {n_gram}')
        if ngram_weights is None:
            ngram_weights = [1.0 / n_gram] * n_gram
        self.ngram_weights = ngram_weights

        # Select tokenizer according to the entered value.
        self.tokenizer_fn = None
        if callable(tokenizer_fn):
            self.tokenizer_fn = tokenizer_fn
        elif isinstance(tokenizer_fn, str):
            self.tokenizer_fn = get_tokenizer(tokenizer_fn)
            if self.tokenizer_fn is None:
                raise ValueError('Right now, `tokenizer_fn` only supports '
                                 "pre-defined 'en' or 'cn'.")
        else:
            assert tokenizer_fn is None, \
                f'`tokenizer_fn` supports Callable, str or None, but not `{type(tokenizer_fn)}`'  # noqa: E501

    def add(self, predictions: Sequence[str], references: Sequence[Sequence[str]]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
             predictions (Sequence[str]): An iterable of predicted sentences.
             references (Sequence[Sequence[str]): An iterable of
                 referenced sentences.
        """
        if self.tokenizer_fn is None:
            language = infer_language(predictions[0])
            self.tokenizer_fn = get_tokenizer(language)
        references_token: Sequence[Sequence[Sequence[str]]] = [
            [self.tokenizer_fn(line) for line in r] for r in references
        ]
        predictions_token: Sequence[Sequence[str]] = [
            self.tokenizer_fn(line) for line in predictions
        ]
        for prediction, references in zip(predictions_token, references_token):
            pred_len = len(prediction)
            # Find the reference that is closest in length to the prediction
            references_len = len(
                min(references, key=lambda x: abs(len(x) - pred_len)))

            pred_counter: Counter = get_n_gram(prediction, self.n_gram)
            reference_counter: Counter = Counter()
            for reference in references:
                # Take union for the n_gram of references.
                reference_counter |= get_n_gram(reference, self.n_gram)

            # Take the intersection of n_gram of prediction and references.
            counter_clip = pred_counter & reference_counter
            precision_matches = np.zeros(self.n_gram)
            precision_total = np.zeros(self.n_gram)
            for counter in counter_clip:
                precision_matches[len(counter) - 1] += counter_clip[counter]
            for counter in pred_counter:
                precision_total[len(counter) - 1] += pred_counter[counter]

            result = (pred_len, references_len, precision_matches,
                      precision_total)
            self._results.append(result)

    def compute_metric(
            self, results: List[Tuple[int, int, np.ndarray,
                                      np.ndarray]]) -> dict:
        """Compute the bleu metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[Tuple[int, int, np.ndarray, np.ndarray]]):
                A list that consisting the tuple of correct numbers.
                Tuple contains pred_len, references_len,
                precision_matches, precision_total.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed bleu score.
        """
        pred_len = 0
        references_len = 0
        precision_matches = np.zeros(self.n_gram)
        precision_total = np.zeros(self.n_gram)
        for result in results:
            pred_len += result[0]
            references_len += result[1]
            precision_matches += result[2]
            precision_total += result[3]

        if min(precision_matches) == 0.0:
            return {'bleu': 0.0}

        if self.smooth:
            precision_score = np.add(precision_matches, np.ones(
                self.n_gram)) / np.add(precision_total, np.ones(self.n_gram))
            precision_score[0] = precision_matches[0] / precision_total[0]
        else:
            precision_score = precision_matches / precision_total

        precision_score = np.array(
            self.ngram_weights) * np.log(precision_score)
        brevity_penalty = _get_brevity_penalty(pred_len, references_len)
        bleu = brevity_penalty * np.exp(np.sum(precision_score))
        result = {'bleu': float(bleu)}
        return result
