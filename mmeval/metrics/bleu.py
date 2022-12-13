# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from mmeval import BaseMetric


def get_n_gram(token: Sequence[str], n_gram: int) -> Counter:
    """A function get n_gram of sentences.

    Args:
        token (Sequence[str]): A series of tokens about sentences.
        n_gram (int): The maximum number of words contained in a phrase
               when calculating word fragments. Defaults to 4.

    Returns:
        Counter: The n_gram contained in sentences with Counter format.
    """
    counter: Counter = Counter()
    for i in range(1, n_gram + 1):
        for j in range(len(token) - i + 1):
            key = tuple(token[j:(i + j)])
            counter[key] += 1
    return counter


def tokenizer_fn(sentence: str) -> Sequence[str]:
    """This function is used to segment a sentence.

    Args:
        sentence(str): A sentence.

    Returns:
        Sequence[str]: A Sequence of tokens after word segmentation.
    """
    return sentence.split()


def _get_brevity_penalty(pred_len: np.array,
                         references_len: np.array) -> np.array:
    """This function is used to calculate penalty factor.

    Args:
        pred_len(np.array): number of grams in the predicted sentence.
        references_len(np.array): number of grams in the references.

    Returns:
        np.array: penalty factor.
    """
    if pred_len > references_len:
        return np.array(1.)
    return np.array(np.exp(1 - references_len / pred_len))


class Bleu(BaseMetric):
    """Bilingual Evaluation Understudy metric.

    This metric is a tool for evaluating the quality of machine translation.
    The closer the translation is to human translation,
    the higher the score will be.

    Args:
        n_gram (int): The maximum number of words contained in a phrase
            when calculating word fragments. Defaults to 4.
        smooth(bool): Whether or not to apply smoothing. Default to False.
        ngram_weights(Sequence[float], optional): Weights used
            for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used. Default to None.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here']  # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']]  # noqa: E501
        >>> bleu = Bleu()
        >>> bleu_results = bleu(predictions, references)
        {'bleu': ...}

    Calculate Bleu with smooth:

        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here']  # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']]  # noqa: E501
        >>> bleu = Bleu(smooth = True)
        >>> bleu_results = bleu(predictions, references)
        {'bleu': ...}
    """

    def __init__(self,
                 n_gram: int = 4,
                 smooth: bool = False,
                 ngram_weights: Optional[Sequence[float]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.smooth = smooth
        if ngram_weights is not None and len(ngram_weights) != n_gram:
            raise ValueError(
                f'List of weights has different weights than `n_gram`: '
                f'{len(ngram_weights)} != {n_gram}')
        if ngram_weights is None:
            ngram_weights = [1.0 / n_gram] * n_gram
        self.ngram_weights = ngram_weights

    def add(self, predictions: Sequence[str], references: Sequence[Sequence[str]]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
             predictions (Sequence[str]): An iterable of machine
                translated corpus.
             references (Sequence[Sequence[str]]): An iterable of
                iterables of reference corpus.
        """

        references_token: Sequence[Sequence[Sequence[str]]] = [[
            tokenizer_fn(line) if line is not None else [] for line in r
        ] for r in references]
        predictions_token: Sequence[Sequence[str]] = [
            tokenizer_fn(line) if line else [] for line in predictions
        ]
        for prediction, references in zip(predictions_token, references_token):
            pred_len = len(prediction)
            references_len = len(min(references, key=lambda x: abs(len(x) - pred_len)))

            pred_counter: Counter = get_n_gram(prediction, self.n_gram)
            reference_counter: Counter = Counter()
            for reference in references:
                # Take intersection for the n_gram of references.
                reference_counter |= get_n_gram(reference, self.n_gram)

            # Union the n_gram of prediction and references.
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
            return {'bleu': np.array(0.0)}

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
