# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import TYPE_CHECKING, Dict, List, Sequence

from mmeval.core import BaseMetric
from mmeval.utils import try_import

if TYPE_CHECKING:
    from rapidfuzz.distance import Levenshtein
else:
    distance = try_import('rapidfuzz.distance')
    if distance is not None:
        Levenshtein = distance.Levenshtein


class OneMinusNormEditDistance(BaseMetric):
    r"""One minus NED metric for text recognition task.

    Args:
        letter_case (str): There are three options to alter the letter cases

            - unchanged: Do not change prediction texts and labels.
            - upper: Convert prediction texts and labels into uppercase
              characters.
            - lower: Convert prediction texts and labels into lowercase
              characters.

            Usually, it only works for English characters. Defaults to
            'unchanged'.
        invalid_symbol (str): A regular expression to filter out invalid or
            not cared characters. Defaults to '[^A-Za-z0-9\u4e00-\u9fa5]'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> from mmeval import OneMinusNormEditDistance
        >>> metric = OneMinusNormEditDistance()
        >>> metric(['helL', 'HEL'], ['hello', 'HELLO'])
        {'1-N.E.D': 0.6}
        >>> metric = OneMinusNormEditDistance(letter_case='upper')
        >>> metric(['helL', 'HEL'], ['hello', 'HELLO'])
        {'1-N.E.D': 0.7}
    """

    def __init__(self,
                 letter_case: str = 'unchanged',
                 invalid_symbol: str = '[^A-Za-z0-9\u4e00-\u9fa5]',
                 **kwargs):
        super().__init__(**kwargs)

        assert letter_case in ['unchanged', 'upper', 'lower']
        self.letter_case = letter_case
        self.invalid_symbol = re.compile(invalid_symbol)

    def add(self, predictions: Sequence[str], groundtruths: Sequence[str]):  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Args:
            predictions (list[str]): The prediction texts.
            groundtruths (list[str]): The ground truth texts.
        """
        for pred, label in zip(predictions, groundtruths):
            if self.letter_case in ['upper', 'lower']:
                pred = getattr(pred, self.letter_case)()
                label = getattr(label, self.letter_case)()
            label = self.invalid_symbol.sub('', label)
            pred = self.invalid_symbol.sub('', pred)
            norm_ed = Levenshtein.normalized_distance(pred, label)
            self._results.append(norm_ed)

    def compute_metric(self, results: List[float]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[float]): The processed results of each batch.

        Returns:
            dict[str, float]: Nested dicts as results.

            - 1-N.E.D (float): One minus the normalized edit distance.
        """
        gt_word_num = len(results)
        norm_ed_sum = sum(results)
        normalized_edit_distance = norm_ed_sum / max(1.0, gt_word_num)
        metric_results = {}
        metric_results['1-N.E.D'] = 1.0 - normalized_edit_distance
        return metric_results
