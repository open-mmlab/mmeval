# Copyright (c) OpenMMLab. All rights reserved.
# This class is modified from `torchmetrics
# <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/rouge.py>`_.
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Sequence, Tuple, Union)

from mmeval import BaseMetric
from mmeval.metrics.utils import get_tokenizer, infer_language
from mmeval.utils import try_import

if TYPE_CHECKING:
    import nltk
else:
    nltk = try_import('nltk')


def _compute_precision_recall_fmeasure(matches: int, pred_len: int,
                                       reference_len: int) -> Dict[str, float]:
    """This computes precision, recall and F1 score based on matches.

    Args:
        matches (int): A number of matches or a length of
            the longest common subsequence.
        pred_len (int): A length of a tokenized predicted sentence.
        reference_len (int): A length of a tokenized referenced sentence.

    Returns:
        Dict[str, float]: A dict with the following keys:

        - precision (float): The precision score.
        - recall (float): The recall score.
        - fmeasure (float): The f1-score.
    """
    if matches == 0:
        return dict(precision=0., recall=0., fmeasure=0.)

    precision = matches / pred_len
    recall = matches / reference_len

    fmeasure = 2 * precision * recall / (precision + recall)
    return dict(
        precision=float(precision),
        recall=float(recall),
        fmeasure=float(fmeasure))


def _rougeL_score(pred: Sequence[str],
                  reference: Sequence[str]) -> Dict[str, float]:
    """This computes precision, recall and F1 score for the Rouge-L metric.

    Args:
        pred (Sequence[str]): A predicted sentence.
        reference (Sequence[str]): A referenced sentence.

    Returns:
        Dict[str, float]: Calculate the score of rougeL.
    """
    pred_len, reference_len = len(pred), len(reference)
    if pred_len == 0 or reference_len == 0:
        return dict(precision=0., recall=0., fmeasure=0.)
    lcs = 0
    matches = SequenceMatcher(None, pred, reference).get_matching_blocks()
    for match in matches:
        lcs += match.size
    return _compute_precision_recall_fmeasure(lcs, pred_len, reference_len)


def _rougeN_score(pred: Sequence[str], reference: Sequence[str],
                  n_gram: int) -> Dict[str, float]:
    """This computes precision, recall and F1 score for the Rouge-N metric.

    Args:
        pred (Sequence[str]): A predicted sentence.
        reference (Sequence[str]): A referenced sentence.
        n_gram (int): The number of words contained in a phrase
            when calculating word fragments.

    Returns:
        Dict[str, float]: Calculate the score of rougeN.
    """

    def _create_ngrams(tokens: Sequence[str], n: int) -> Counter:
        ngrams: Counter = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams

    pred_ngarms = _create_ngrams(pred, n_gram)
    reference_ngarms = _create_ngrams(reference, n_gram)
    pred_len = sum(pred_ngarms.values())
    reference_len = sum(reference_ngarms.values())
    if pred_len == 0 or reference_len == 0:
        return dict(precision=0., recall=0., fmeasure=0.)

    # Take the intersection of n_gram of prediction and reference.
    hits = sum(
        min(pred_ngarms[w], reference_ngarms[w]) for w in set(pred_ngarms))
    return _compute_precision_recall_fmeasure(hits, pred_len, reference_len)


class ROUGE(BaseMetric):
    """Calculate Rouge Score used for automatic summarization.

    This metric proposed in `ROUGE: A Package for Automatic Evaluation
    of Summaries <https://www.aclweb.org/anthology/W04-1013.pdf>`_ are
    common evaluation indicators in the fields of machine translation,
    automatic summarization, question and answer generation, etc.

    Args:
        rouge_keys (List or Tuple or int or str):
            A list of rouge types to calculate.
            Keys that are allowed are ``L``, and ``1`` through ``9``.
            Defaults to ``(1, 2, 'L')``.
        use_stemmer (bool): Use Porter stemmer to strip word
            suffixes to improve matching. Defaults to False.
        normalizer (Callable, optional): A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters
            with spaces is default. Defaults to None.
        tokenizer (Callable or str, optional): A user's own tokenizer function.
            Defaults to None.
        accumulate (str): Useful in case of multi-reference rouge score.
            ``avg`` takes the average of all references with respect to
            predictions. ``best`` takes the best fmeasure score obtained
            between prediction and multiple corresponding references.
            Defaults to ``best``.
        lowercase (bool): If it is True, all characters will be lowercase.
            Defaults to True.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import ROUGE
        >>> predictions = ['the cat is on the mat']
        >>> references = [['a cat is on the mat']]
        >>> metric = ROUGE(rouge_keys='L')
        >>> metric.add(predictions, references)
        >>> results = metric.compute_metric()
        {'rougeL_fmeasure': 0.8333333,
         'rougeL_precision': 0.8333333,
         'rougeL_recall': 0.8333333}
    """

    def __init__(self,
                 rouge_keys: Union[List, Tuple, int, str] = (1, 2, 'L'),
                 use_stemmer: bool = False,
                 normalizer: Optional[Callable] = None,
                 tokenizer: Union[Callable, str, None] = None,
                 accumulate: str = 'best',
                 lowercase: bool = True,
                 **kwargs: Any):
        super().__init__(**kwargs)
        if isinstance(rouge_keys, int) or isinstance(rouge_keys, str):
            rouge_keys = [rouge_keys]
        # Check the legitimacy of the rouge_keys
        for rouge_key in rouge_keys:
            if isinstance(rouge_key, int):
                if rouge_key < 1 or rouge_key > 9:
                    raise ValueError(f'Got unknown rouge key {rouge_key}. '
                                     'Expected to be one of {1 - 9} or L')
            elif rouge_key != 'L':
                raise ValueError(f'Got unknown rouge key {rouge_key}. '
                                 'Expected to be one of {1 - 9} or L')
        self.rouge_keys = rouge_keys

        # use stemmer in nltk if necessary
        if use_stemmer and nltk is not None:
            self.stemmer = nltk.stem.porter.PorterStemmer()
        elif use_stemmer and nltk is None:
            raise ValueError(
                'The nltk package is needed to use stemmer, '
                'check https://www.nltk.org/install.html for installation.')
        else:
            self.stemmer = None
        self.normalizer = normalizer

        # Select tokenizer according to the entered value.
        self.tokenizer_fn = None
        if callable(tokenizer):
            self.tokenizer_fn = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer_fn = get_tokenizer(tokenizer)
            if self.tokenizer_fn is None:
                raise ValueError('Right now, `tokenizer` only supports '
                                 "pre-defined 'en' or 'cn'.")
        else:
            assert tokenizer is None, \
                f'`tokenizer` supports Callable, str or None, but not `{type(tokenizer)}`'  # noqa: E501
        assert accumulate in ['best', 'avg'], \
            f'Wrong accumulate {accumulate}. Supported accumulate are "best" and "avg"'  # noqa: E501
        self.accumulate = accumulate
        self.lowercase = lowercase

    def add(self, predictions: Sequence[str], references: Sequence[Sequence[str]]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
             predictions (Sequence[str]): An iterable of predicted sentences.
             references (Sequence[Sequence[str]): An iterable of
                 referenced sentences. Each predicted sentence may
                 correspond to multiple referenced sentences.
        """
        # If the tokenizer is None, check the first sentence
        # to determine which language the tokenizer is used.
        if self.tokenizer_fn is None:
            language = infer_language(predictions[0])
            self.tokenizer_fn = get_tokenizer(language)

        # Traverse the predicted sentences
        for prediction, _references in zip(predictions, references):
            scores_per_rouge_keys = self._compute_rouge_score(
                prediction, _references)
            self._results.append(scores_per_rouge_keys)

    def _compute_rouge_score(self, prediction: str,
                             references: Sequence[str]) -> Sequence[tuple]:
        """Compute the rouge score.

        Args:
            prediction (str): The predicted sentence.
            references (Sequence[str]): The referenced sentences.
                Each predicted sentence may correspond to multiple
                referenced sentences.

        Returns:
            Sequence[tuple]: The rouge scores corresponding to each
            ``rouge_key``. And each scores is a tuple of
            (fmeasure, precision, recall).
        """
        assert isinstance(references, Sequence), \
            f'The `references` should be a sequence of string, but got {type(references)}.'  # noqa: E501
        assert len(references) > 0, \
            'The number of references should large than 0.'

        pred_token = self._normalize_and_tokenize(prediction)
        ref_tokens = [
            self._normalize_and_tokenize(refs) for refs in references
        ]

        # Traverse the chosen rouge_keys
        scores_per_rouge_keys = []
        for rouge_key in self.rouge_keys:

            # Traverse the tokens of references for single prediction.
            scores = []
            for ref_token in ref_tokens:
                if isinstance(rouge_key, int):
                    score = _rougeN_score(pred_token, ref_token, rouge_key)
                else:
                    score = _rougeL_score(pred_token, ref_token)
                scores.append(score)

            # Accumulate rouge score across multiple reference.
            if self.accumulate == 'best':
                fmeasure = max(score['fmeasure'] for score in scores)
                precision = max(score['precision'] for score in scores)
                recall = max(score['recall'] for score in scores)
            else:
                fmeasure = sum(score['fmeasure']
                               for score in scores) / len(scores)
                precision = sum(score['precision']
                                for score in scores) / len(scores)
                recall = sum(score['recall'] for score in scores) / len(scores)

            scores_per_rouge_keys.append((fmeasure, precision, recall))

        return scores_per_rouge_keys

    def compute_metric(self, results: List[Any]) -> dict:
        """Compute the rouge metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List): A list that consists correct numbers.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed rouge score.
        """
        fmeasure = [0] * len(self.rouge_keys)
        recall = [0] * len(self.rouge_keys)
        precision = [0] * len(self.rouge_keys)
        for result in results:
            for i, each_rouge in enumerate(result):
                fmeasure[i] += each_rouge[0]
                precision[i] += each_rouge[1]
                recall[i] += each_rouge[2]

        metric_results = {}
        num_samples = len(self._results)
        for i, rouge_key in enumerate(self.rouge_keys):
            metric_results[
                f'rouge{rouge_key}_fmeasure'] = fmeasure[i] / num_samples
            metric_results[
                f'rouge{rouge_key}_precision'] = precision[i] / num_samples
            metric_results[
                f'rouge{rouge_key}_recall'] = recall[i] / num_samples
        return metric_results

    def _normalize_and_tokenize(self, text: str) -> Sequence[str]:
        """Normalize and tokenize the given text.

        Rouge score should be calculated only over lowercased words and
        digits. Optionally, ``nltk.stem.porter.PorterStemmer`` can be used
        to strip word suffixes for better matching.

        Args:
            text (str): An input sentence.

        Returns:
            Sequence[str]: The tokens after normalizer and tokenizer.
        """
        if self.tokenizer_fn == str.split:
            if callable(self.normalizer):
                text = self.normalizer(text)
            elif self.lowercase:
                text = re.sub(r'[^a-z0-9]+', ' ', text.lower())
            else:
                text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
        tokens = self.tokenizer_fn(text)  # type: ignore
        if self.stemmer:
            tokens = [
                self.stemmer.stem(x) if len(x) > 3 else x for x in tokens
            ]
        tokens = [x for x in tokens if (isinstance(x, str) and len(x) > 0)]
        return tokens
