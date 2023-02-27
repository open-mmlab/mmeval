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
        valid_symbol (str): Valid characters. Defaults to
            '[^A-Z^a-z^0-9^\u4e00-\u9fa5]'.

    Example:
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
                 valid_symbol: str = '[^A-Z^a-z^0-9^\u4e00-\u9fa5]',
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.valid_symbol = re.compile(valid_symbol)
        assert isinstance(mode, (str, list))
        if isinstance(mode, str):
            mode = [mode]
        assert all([isinstance(item, str) for item in mode])
        assert set(mode).issubset(
            {'exact', 'ignore_case', 'ignore_case_symbol'})
        self.mode = set(mode)  # type: ignore

    def add(self, predictions: Sequence[str], labels: Sequence[str]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Args:
            predictions (list[str]): The prediction texts.
            labels (list[str]): The ground truth texts.
        """
        for pred, label in zip(predictions, labels):
            num, ignore_case_num, ignore_case_symbol_num = 0, 0, 0
            if 'exact' in self.mode:
                num = pred == label
            if 'ignore_case' in self.mode or 'ignore_case_symbol' in self.mode:
                pred_lower = pred.lower()
                label_lower = label.lower()
                ignore_case_num = pred_lower == label_lower
            if 'ignore_case_symbol' in self.mode:
                label_lower_ignore = self.valid_symbol.sub('', label_lower)
                pred_lower_ignore = self.valid_symbol.sub('', pred_lower)
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
        eval_res = {}
        gt_word_num = max(len(results), 1.0)
        exact_sum, ignore_case_sum, ignore_case_symbol_sum = 0.0, 0.0, 0.0
        for exact, ignore_case, ignore_case_symbol in results:
            exact_sum += exact
            ignore_case_sum += ignore_case
            ignore_case_symbol_sum += ignore_case_symbol
        if 'exact' in self.mode:
            eval_res['accuracy'] = exact_sum / gt_word_num
        if 'ignore_case' in self.mode:
            eval_res['ignore_case_accuracy'] = ignore_case_sum / gt_word_num
        if 'ignore_case_symbol' in self.mode:
            eval_res['ignore_case_symbol_accuracy'] =\
                ignore_case_symbol_sum / gt_word_num
        return eval_res
