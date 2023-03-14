# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from collections import OrderedDict
from multiprocessing.pool import Pool
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics.utils import calculate_overlaps
from mmeval.utils import is_list_of


class ProposalRecall(BaseMetric):
    """Proposals recall evaluation metric.

    The speed of calculating recall is faster than COCO Detection metric.

    Args:
        iou_thrs (float | List[float], optional): IoU thresholds.
            If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (1, 10, 100, 1000).
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        print_results (bool): Whether to print the results. Defaults to True.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> import numpy as np
        >>> from mmeval import ProposalRecall
        >>>
        >>> proposal_recall = ProposalRecall()
        >>> def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
        ...     # random generate bounding boxes in 'xyxy' formart.
        ...     x = np.random.rand(num_bboxes, ) * img_w
        ...     y = np.random.rand(num_bboxes, ) * img_h
        ...     w = np.random.rand(num_bboxes, ) * (img_w - x)
        ...     h = np.random.rand(num_bboxes, ) * (img_h - y)
        ...     return np.stack([x, y, x + w, y + h], axis=1)
        >>>
        >>> prediction = {
        ...     'bboxes': _gen_bboxes(10),
        ...     'scores': np.random.rand(10, ),
        ... }
        >>> groundtruth = {
        ...     'bboxes': _gen_bboxes(10),
        ... }
        >>> proposal_recall(predictions=[prediction, ], groundtruths=[groundtruth, ])  # doctest: +ELLIPSIS  # noqa: E501
        {'AR@1': ..., 'AR@10': ..., 'AR@100': ..., 'AR@1000': ...}
    """

    def __init__(self,
                 iou_thrs: Union[float, Sequence[float], None] = None,
                 proposal_nums: Union[int, Sequence[int]] = (1, 10, 100, 1000),
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 print_results: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        elif isinstance(iou_thrs, float):
            iou_thrs = np.array([iou_thrs])
        elif is_list_of(iou_thrs, float):
            iou_thrs = np.array(iou_thrs)
        else:
            raise TypeError(
                '`iou_thrs` should be None, float, or a list of float, '
                f'but got {iou_thrs}')
        self.iou_thrs = iou_thrs

        if isinstance(proposal_nums, (list, tuple)):
            proposal_nums = np.array(proposal_nums)
        elif isinstance(proposal_nums, int):
            proposal_nums = np.array([proposal_nums])
        else:
            raise TypeError('proposal_nums should be int or Sequence[int],'
                            f'but got {type(proposal_nums)}')
        self.proposal_nums = proposal_nums

        self.nproc = nproc
        self.use_legacy_coordinate = use_legacy_coordinate
        self.print_results = print_results

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - bboxes (numpy.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, 1), the predicted scores
                  of bounding boxes.

            groundtruths (Sequence[dict]): A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:

                - bboxes (numpy.ndarray): Shape (M, 4), the ground truth
                  bounding bboxes of this image, in 'xyxy' foramrt.
        """
        for prediction, label in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(label, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(label)}.'
            self._results.append((prediction, label))

    def process_proposals(self, predictions: List[dict],
                          groundtruths: List[dict]) -> Tuple[List, List, int]:
        """Process the proposals(bboxes) in predictions and groundtruths.

        Args:
             predictions (List[dict]): A list of dict. Each dict is the
                detection result of an image. Same as
                :class:`ProposalRecall.add`.
             groundtruths (list[dict]): Same as :class:`ProposalRecall.add`.

        Returns:
            tuple (proposal_preds, proposal_gts, num_gts):

            - proposal_preds (List[numpy.ndarray]): The sorted proposals of
              the prediction.
            - proposal_gts (List[numpy.ndarray]): The proposals of the
              groundtruths.
            - num_gts (int): The total groundtruth numbers.
        """
        proposal_preds = []
        proposal_gts = []
        num_gts = 0

        for prediction, groundtruth in zip(predictions, groundtruths):
            # process prediction
            proposal_pred = prediction['bboxes']
            scores = prediction.get('scores', None)
            if scores is not None:
                sort_idx = np.argsort(scores)[::-1]
                proposal_pred = proposal_pred[sort_idx, :]
            prop_num = min(proposal_pred.shape[0],
                           self.proposal_nums[-1])  # type:ignore
            proposal_pred = proposal_pred[:prop_num, :]

            # process groundtruth
            proposal_gt = groundtruth.get('bboxes')
            if proposal_gt is None:
                proposal_gt = np.zeros((0, 4), dtype=np.float32)

            proposal_preds.append(proposal_pred)
            proposal_gts.append(proposal_gt)
            num_gts += proposal_gt.shape[0]
        return proposal_preds, proposal_gts, num_gts

    def calculate_recall(self, predictions: List[dict],
                         groundtruths: List[dict], pool: Optional[Pool]):
        """Calculate recall.

        Args:
            predictions (List[dict]): A list of dict. Each dict is the
                detection result of an image. Same as
                :class:`ProposalRecall.add`.
            groundtruths (List[dict]): A list of dict. Each dict is the
                ground truth of an image. Same as
                :class:`ProposalRecall.add`.
            pool (Optional[Pool]): A instance of :class:`multiprocessing.Pool`.
                If None, do not use multiprocessing.

        Returns:
            numpy.ndarry: Shape (len(proposal_nums), len(iou_thrs)),
            the recall results.
        """
        proposal_preds, proposal_gts, num_gts = self.process_proposals(
            predictions=predictions, groundtruths=groundtruths)

        if pool is not None:
            num_images = len(proposal_preds)
            ious = pool.starmap(
                self._calculate_ious,
                zip(proposal_preds, proposal_gts,
                    [self.proposal_nums] * num_images,
                    [self.use_legacy_coordinate] * num_images))
        else:
            ious = []
            for img_idx in range(len(proposal_preds)):
                _iou = self._calculate_ious(proposal_preds[img_idx],
                                            proposal_gts[img_idx],
                                            self.proposal_nums,
                                            self.use_legacy_coordinate)
                ious.append(_iou)

        ious = np.concatenate(ious, axis=1)
        ious = np.fliplr(np.sort(ious, axis=1))
        recalls = np.zeros((
            len(self.proposal_nums),  # type:ignore
            len(self.iou_thrs)))  # type:ignore
        for i, thr in enumerate(self.iou_thrs):  # type:ignore
            recalls[:, i] = (ious >= thr).sum(axis=1) / float(num_gts)
        return recalls

    @staticmethod
    def _calculate_ious(pred_proposals: np.ndarray, gt_proposals: np.ndarray,
                        proposal_nums: np.ndarray,
                        use_legacy_coordinate: bool):
        """Calculate the IoUs under different proposal numbers on an image.

        Args:
            pred_proposals (numpy.ndarray): Predicted proposals of this
                image, with shape (M, 4).
            gt_proposals (numpy.ndarray): Ground truth proposals of this
                image, with shape (M, 4).
            proposal_nums (numpy.ndarry): Numbers of proposals to be evaluated.
            use_legacy_coordinate (bool): Refer to :class:`ProposalRecall`.

        Returns:
            numpy.ndarray: Shape (len(proposal_nums), N),
            IoUs under different proposal numbers on an image.

        Note:
            This method should be a staticmethod to avoid resource competition
            during multiple processes.
        """
        # Step 1. calculate all proposal ious
        if gt_proposals is None or gt_proposals.shape[0] == 0:
            ious = np.zeros((0, pred_proposals.shape[0]), dtype=np.float32)
        else:
            ious = calculate_overlaps(
                gt_proposals,
                pred_proposals,
                use_legacy_coordinate=use_legacy_coordinate)
        # Step 2. initialize the ious array and calculate based on proposal_num
        _ious = np.zeros((proposal_nums.size, gt_proposals.shape[0]),
                         dtype=np.float32)  # type:ignore
        for i, proposal_num in enumerate(proposal_nums):
            pred_ious = ious[:, :proposal_num].copy()
            gt_ious = np.zeros(ious.shape[0])

            if pred_ious.size == 0:
                _ious[i, :] = gt_ious
                continue
            for j in range(pred_ious.shape[0]):
                gt_max_overlaps = pred_ious.argmax(axis=1)
                max_ious = pred_ious[np.arange(0, pred_ious.shape[0]),
                                     gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                pred_ious[gt_idx, :] = -1
                pred_ious[:, box_idx] = -1
            _ious[i, :] = gt_ious
        return _ious

    def compute_metric(self, results: list) -> dict:
        """Compute the ProposalRecall metric.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric. The keys are the names of
            the proposal numbers, and the values are corresponding results.
        """
        predictions, groundtruths = zip(*results)

        nproc = min(self.nproc, len(predictions))
        if nproc > 1:
            pool = Pool(nproc)
        else:
            pool = None  # type: ignore

        recalls = self.calculate_recall(predictions, groundtruths, pool)
        ar = recalls.mean(axis=1)
        if pool is not None:
            pool.close()

        eval_results: OrderedDict = OrderedDict()
        table_results: OrderedDict = OrderedDict()
        results_list = []
        for i, num in enumerate(self.proposal_nums):  # type:ignore
            eval_results[f'AR@{num}'] = ar[i]
            results_list.append(np.concatenate([recalls[i], ar[None, i]]))
        table_results['proposal_result'] = results_list

        if self.print_results:
            self._print_results(table_results)
        return eval_results

    def _print_results(self, table_results: dict) -> None:
        """Print the evaluation results table.

        Args:
            table_results (dict): The computed metric.
        """
        tabel_title = ' Recall Results (%)'
        result = table_results['proposal_result']
        headers = [''] + [
            f'AR_{round(iou_thr * 100, 0):.0f}'
            for iou_thr in self.iou_thrs  # type: ignore
        ] + ['AR']

        table = Table(title=tabel_title)
        console = Console(width=150)
        for name in headers:
            table.add_column(name, justify='left')

        for i in range(len(self.proposal_nums)):  # type: ignore
            row = [f'{self.proposal_nums[i]}']  # type: ignore
            row += [
                f'{round(100 * val, 2):0.2f}' for val in result[i].tolist()
            ]
            table.add_row(*row)

        with console.capture() as capture:
            console.print(table, end='')
        self.logger.info('\n' + capture.get())
