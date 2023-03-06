# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Dict, List, Sequence, Tuple, Union

from mmeval.core import BaseMetric


class WordAccuracy(BaseMetric):
    r"""Calculate the word level accuracy.

    Args:
        mode (str or list[str]): Options are:

            - 'exact': Accuracy at word level.
            - 'ignore_case': Accuracy at word level, ignoring letter
              case.
            - 'ignore_case_symbol': Accuracy at word level, ignoring
              letter case and symbol. (Default metric for academic evaluation)

            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'ignore_case_symbol'.
        invalid_symbol (str): A regular expression to filter out invalid or
            not cared characters. Defaults to '[^A-Za-z0-9\u4e00-\u9fa5]'
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> from mmeval import WordAccuracy
        >>> metric = WordAccuracy()
        >>> metric(['hello', 'hello', 'hello'], ['hello', 'HELLO', '$HELLO$'])
        {'ignore_case_symbol_accuracy': 1.0}
        >>> metric = WordAccuracy(mode=['exact', 'ignore_case',
        >>>                             'ignore_case_symbol'])
        >>> metric(['hello', 'hello', 'hello'], ['hello', 'HELLO', '$HELLO$'])
        {'accuracy': 0.333333333,
         'ignore_case_accuracy': 0.666666667,
         'ignore_case_symbol_accuracy': 1.0}
    """

    def __init__(self,
                 mode: Union[str, Sequence[str]] = 'ignore_case_symbol',
                 invalid_symbol: str = '[^A-Za-z0-9\u4e00-\u9fa5]',
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.invalid_symbol = re.compile(invalid_symbol)
        assert isinstance(mode, (str, list))
        if isinstance(mode, str):
            mode = [mode]
        assert all(isinstance(item, str) for item in mode)
        self.mode = set(mode)  # type: ignore
        assert set(self.mode).issubset(
            {'exact', 'ignore_case', 'ignore_case_symbol'})

    def add(self, predictions: Sequence[str], groundtruths: Sequence[str]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Args:
            predictions (list[str]): The prediction texts.
            groundtruths (list[str]): The ground truth texts.
        """
        for pred, label in zip(predictions, groundtruths):
            num, ignore_case_num, ignore_case_symbol_num = 0, 0, 0
            if 'exact' in self.mode:
                num = pred == label
            if 'ignore_case' in self.mode or 'ignore_case_symbol' in self.mode:
                pred_lower = pred.lower()
                label_lower = label.lower()
                ignore_case_num = pred_lower == label_lower
            if 'ignore_case_symbol' in self.mode:
                label_lower_ignore = self.invalid_symbol.sub('', label_lower)
                pred_lower_ignore = self.invalid_symbol.sub('', pred_lower)
                ignore_case_symbol_num =\
                    label_lower_ignore == pred_lower_ignore
            self._results.append(
                (num, ignore_case_num, ignore_case_symbol_num))

    def compute_metric(self, results: List[Tuple[int, int, int]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[float]): The processed results of each batch.

        Returns:
            dict[str, float]: Nested dicts as results. Provided keys are:

            - accuracy (float): Accuracy at word level.
            - ignore_case_accuracy (float): Accuracy at word level, ignoring
              letter case.
            - ignore_case_symbol_accuracy (float): Accuracy at word level,
              ignoring letter case and symbol.
        """
        metric_results = {}
        gt_word_num = max(len(results), 1.0)
        exact_sum, ignore_case_sum, ignore_case_symbol_sum = 0.0, 0.0, 0.0
        for exact, ignore_case, ignore_case_symbol in results:
            exact_sum += exact
            ignore_case_sum += ignore_case
            ignore_case_symbol_sum += ignore_case_symbol
        if 'exact' in self.mode:
            metric_results['accuracy'] = exact_sum / gt_word_num
        if 'ignore_case' in self.mode:
            metric_results[
                'ignore_case_accuracy'] = ignore_case_sum / gt_word_num
        if 'ignore_case_symbol' in self.mode:
            metric_results['ignore_case_symbol_accuracy'] =\
                ignore_case_symbol_sum / gt_word_num
        return metric_results
