# Copyright (c) OpenMMLab. All rights reserved.
import re
from difflib import SequenceMatcher
from typing import Dict, Sequence, Tuple

from mmeval.core import BaseMetric


class CharRecallPrecision(BaseMetric):
    r"""Calculate the char level recall & precision.

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
        >>> from mmeval import CharRecallPrecision
        >>> metric = CharRecallPrecision()
        >>> metric(['helL', 'HEL'], ['hello', 'HELLO'])
        {'char_recall': 0.6, 'char_precision': 0.8571428571428571}
        >>> metric = CharRecallPrecision(letter_case='upper')
        >>> metric(['helL', 'HEL'], ['hello', 'HELLO'])
        {'char_recall': 0.7, 'char_precision': 1.0}
    """

    def __init__(self,
                 letter_case: str = 'unchanged',
                 invalid_symbol: str = '[^A-Za-z0-9\u4e00-\u9fa5]',
                 **kwargs):
        super().__init__(**kwargs)
        assert letter_case in ['unchanged', 'upper', 'lower']
        self.letter_case = letter_case
        self.invalid_symbol = re.compile(invalid_symbol)

    def add(self, predictions: Sequence[str], groundtruths: Sequence[str]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Args:
            predictions (list[str]): The prediction texts.
            groundtruths (list[str]): The ground truth texts.
        """
        for pred, label in zip(predictions, groundtruths):
            if self.letter_case in ['upper', 'lower']:
                pred = getattr(pred, self.letter_case)()
                label = getattr(label, self.letter_case)()
            valid_label = self.invalid_symbol.sub('', label)
            valid_pred = self.invalid_symbol.sub('', pred)
            # number to calculate char level recall & precision
            true_positive_char_num = self._cal_true_positive_char(
                valid_pred, valid_label)
            self._results.append(
                (len(valid_label), len(valid_pred), true_positive_char_num))

    def compute_metric(self, results: Sequence[Tuple[int, int, int]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[tuple]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the
            metrics, and the values are corresponding results.
        """
        gt_sum, pred_sum, true_positive_sum = 0.0, 0.0, 0.0
        for gt, pred, true_positive in results:
            gt_sum += gt
            pred_sum += pred
            true_positive_sum += true_positive
        char_recall = true_positive_sum / max(gt_sum, 1.0)
        char_precision = true_positive_sum / max(pred_sum, 1.0)
        metric_results = {}
        metric_results['recall'] = char_recall
        metric_results['precision'] = char_precision
        return metric_results

    def _cal_true_positive_char(self, pred: str, gt: str) -> int:
        """Calculate correct character number in prediction.

        Args:
            pred (str): Prediction text.
            gt (str): Ground truth text.

        Returns:
            true_positive_char_num (int): The true positive number.
        """

        all_opt = SequenceMatcher(None, pred, gt)
        true_positive_char_num = 0
        for opt, _, _, s2, e2 in all_opt.get_opcodes():
            if opt == 'equal':
                true_positive_char_num += (e2 - s2)
            else:
                pass
        return true_positive_char_num
