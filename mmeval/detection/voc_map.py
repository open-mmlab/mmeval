# Copyright (c) OpenMMLab. All rights reserved.

import multiprocessing
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from mmeval.detection.utils.bbox import (calculate_bboxes_area,
                                         calculate_overlaps)


class VOCMeanAP(BaseMetric):
    """Pascal VOC evaluation metric.

    Args:
        iou_thrs (float or Tuple[float], optional): IoU threshold.
            Defaults to 0.5.
        scale_ranges (Tuple[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        eval_mode (str, optional): 'area' or '11points', 'area' means
            calculating the area under precision-recall curve, '11points' means
            calculating the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.
    """

    def __init__(self,
                 iou_thrs: Union[float, Tuple[float]] = 0.5,
                 scale_ranges: Optional[List[Tuple]] = None,
                 eval_mode: str = '11points',
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(iou_thrs, float):
            iou_thrs = (iou_thrs, )
        self.iou_thrs = iou_thrs

        if scale_ranges is None:
            scale_ranges = [(None, None)]
        self.scale_ranges = scale_ranges

        area_ranges = []
        for min_scale, max_scale in self.scale_ranges:
            min_area = min_scale if min_scale is None else min_scale**2
            max_area = max_scale if max_scale is None else max_scale**2
            area_ranges.append((min_area, max_area))
        self._area_ranges = area_ranges

        assert eval_mode in ['area', '11points'], \
            'Unrecognized mode, only "area" and "11points" are supported'
        self.eval_mode = eval_mode

        self.nproc = nproc
        self.use_legacy_coordinate = use_legacy_coordinate

    def add(self, predictions: List[Dict], labels: List[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): Predictions from the model. With the
                following keys:
                    - bboxes
                    - scores
                    - labels
            labels (Sequence[dict]): The ground truth labels. With the
                following keys:
                    - bboxes
                    - labels
                    - bboxes_ignore
                    - labels_ignore
        """
        for prediction, label in zip(predictions, labels):
            self._results.append((prediction, label))

    def _calculate_average_precision(self, precisions: np.ndarray,
                                     recalls: np.ndarray) -> np.ndarray:
        """Calculate average precision (for multiple scales and multiple ious).

        Args:
            recalls (ndarray): shape (num_ious, num_scales, num_dets)
            precisions (ndarray): shape (num_ious, num_scales, num_dets).
            mode (str): 'area' or '11points', 'area' means calculating the area
                under precision-recall curve, '11points' means calculating
                the average precision of recalls at [0, 0.1, ..., 1]

        Returns:
            numpy.ndarray: calculated average precision
        """
        num_scales = len(self.scale_ranges)
        num_iou_thrs = len(self.iou_thrs)
        ap = np.zeros((num_iou_thrs, num_scales), dtype=np.float32)
        if self.eval_mode == 'area':
            raise NotImplementedError
        elif self.eval_mode == '11points':
            for i in range(num_iou_thrs):
                for j in range(num_scales):
                    for thr in np.arange(0, 1 + 1e-3, 0.1):
                        precs = precisions[i, j, recalls[i, j, :] >= thr]
                        prec = precs.max() if precs.size > 0 else 0
                        ap[i, j] += prec
            ap /= 11
        else:
            raise ValueError(
                'Unrecognized mode, only "area" and "11points" are supported')

        return ap

    def _filter_by_bboxes_area(self, bboxes: np.ndarray,
                               min_area: Optional[float],
                               max_area: Optional[float]) -> np.ndarray:
        """Filter the bboxes area."""
        bboxes_area = calculate_bboxes_area(bboxes, self.use_legacy_coordinate)
        area_mask = np.ones_like(bboxes_area, dtype=bool)
        if min_area is not None:
            area_mask &= (bboxes_area >= min_area)
        if max_area is not None:
            area_mask &= (bboxes_area < max_area)
        return area_mask

    def _calculate_tpfp(
            self, pred_bboxes: np.ndarray,
            gt_bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Check if detected bboxes are true positive or false positive.

        Args:
            pred_bboxes (numpy.ndarray): Detected bboxes of this image, of
                shape (m, 5).
            gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).

        Returns:
            tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape
            of each array is (num_ious, num_scales, m).
        """
        # Step 1. Initialize the `tp` and `fp` arrays.
        num_preds = pred_bboxes.shape[0]
        num_iou_thrs = len(self.iou_thrs)
        num_scales = len(self.scale_ranges)
        tp = np.zeros((num_iou_thrs, num_scales, num_preds), dtype=np.float32)
        fp = np.zeros((num_iou_thrs, num_scales, num_preds), dtype=np.float32)

        # Step 2. If there is no gt bboxes in this image, then all pred bboxes
        # within area range are false positives.
        if gt_bboxes.shape[0] == 0:
            for idx, (min_area, max_area) in enumerate(self._area_ranges):
                area_mask = self._filter_by_bboxes_area(
                    pred_bboxes[:, :4], min_area, max_area)
                fp[:, idx, area_mask] = 1
            return tp, fp

        # Step 3. Calculate the IoUs between the predicted bboxes and the
        # ground truth bboxes.
        ious = calculate_overlaps(
            pred_bboxes[:, :4],
            gt_bboxes[:, :4],
            mode='iou',
            use_legacy_coordinate=self.use_legacy_coordinate)
        # For each pred bbox, the max iou with all gts.
        ious_max = ious.max(axis=1)
        # For each pred bbox, which gt overlaps most with it.
        ious_argmax = ious.argmax(axis=1)
        # Sort all pred bbox in descending order by scores.
        sorted_indices = np.argsort(-pred_bboxes[:, -1])

        # Step 4. Count the `tp` and `fp` of each iou threshold and area range.
        for iou_thr_idx, iou_thr in enumerate(self.iou_thrs):
            for area_idx, (min_area, max_area) in enumerate(self._area_ranges):
                # The flags that gt bboxes have been matched.
                gt_covered_flags = np.zeros(gt_bboxes.shape[0], dtype=bool)
                # The flags that gt bboxes should be ignored.
                ignore_gt_flags = gt_bboxes[:, -1].astype(bool)
                # The flags that gt bboxes out of area range.
                gt_area_mask = self._filter_by_bboxes_area(
                    gt_bboxes, min_area, max_area)
                ignore_gt_area_flags = ~gt_area_mask

                # Count the prediction bboxes in order of decreasing score.
                for pred_bbox_idx in sorted_indices:
                    if ious_max[pred_bbox_idx] >= iou_thr:
                        matched_gt_idx = ious_argmax[pred_bbox_idx]
                        # Ignore the pred bbox that match an ignored gt bbox.
                        if ignore_gt_flags[matched_gt_idx]:
                            continue
                        # Ignore the pred bbox that is out of area range.
                        if ignore_gt_area_flags[matched_gt_idx]:
                            continue
                        if not gt_covered_flags[matched_gt_idx]:
                            tp[iou_thr_idx, area_idx, pred_bbox_idx] = 1
                            gt_covered_flags[matched_gt_idx] = True
                        else:
                            # This gt bbox has been matched and counted as fp.
                            fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1
                    else:
                        area_mask = self._filter_by_bboxes_area(
                            pred_bboxes[pred_bbox_idx, :4], min_area, max_area)
                        if area_mask:
                            fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1

        return tp, fp

    def get_class_results(self, predictions: List[dict],
                          groundtruths: List[dict], class_index: int) -> Tuple:
        """Get prediciton results and gt information of a certain class.

        Args:
            predictions (list[dict]): Same as ``self.add``.
            groundtruths (list[dict]): Same as ``self.add``.
            class_index (int): Index of a specific class.

        Returns:
            tuple[list[np.ndarray]]: predictions bboxes, gt bboxes.
        """
        class_preds = []
        class_gts = []

        for pred, gt in zip(predictions, groundtruths):
            pred_indices = (pred['labels'] == class_index)
            pred_bboxes_info = np.hstack([
                pred['bboxes'][pred_indices, :],
                pred['scores'][pred_indices].reshape((-1, 1))
            ])
            class_preds.append(pred_bboxes_info)

            gt_indices = (gt['labels'] == class_index)
            gt_bboxes = gt['bboxes'][gt_indices, :]
            gt_bboxes_info = np.hstack(
                [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))])

            if gt.get('labels_ignore', None) is not None:
                ignore_gt_indices = (gt['labels_ignore'] == class_index)
                ignore_gt_bboxes = gt['bboxes_ignore'][ignore_gt_indices, :]
                ignore_gt_bboxes_info = np.hstack([
                    ignore_gt_bboxes,
                    np.ones((ignore_gt_bboxes.shape[0], 1))
                ])
                gt_bboxes_info = np.vstack(
                    (gt_bboxes_info, ignore_gt_bboxes_info))

            class_gts.append(gt_bboxes_info)

        return class_preds, class_gts

    def compute_metric(self, results: list) -> dict:
        """Compute the VOCMeanAP metric.

        Args:
            results (list): TODO.

        Returns:
            dict: The computed metric.
        """
        predictions, groundtruths = zip(*results)

        class_names = self.dataset_meta['CLASSES']  # type: ignore
        num_images = len(predictions)

        nproc = min(self.nproc, num_images)
        if nproc > 1:
            pool = multiprocessing.Pool(nproc)
        else:
            pool = None  # type: ignore

        results_per_class = []
        for class_index in range(len(class_names)):
            class_preds, class_gts = self.get_class_results(
                predictions, groundtruths, class_index)

            if pool is not None:
                tpfp_list = pool.starmap(self._calculate_tpfp,
                                         zip(class_preds, class_gts))
            else:
                tpfp_list = [
                    self._calculate_tpfp(class_preds[i], class_gts[i])
                    for i in range(num_images)
                ]

            image_tp_list, image_fp_list = tuple(zip(*tpfp_list))

            num_gts = np.zeros((len(self.iou_thrs), len(self.scale_ranges)),
                               dtype=int)

            for image_idx in range(num_images):
                for idx, (min_area, max_area) in enumerate(self._area_ranges):
                    area_mask = self._filter_by_bboxes_area(
                        class_gts[image_idx], min_area, max_area)
                    num_gts[:, idx] = np.sum(area_mask)

            sorted_indices = np.argsort(-np.vstack(class_preds)[:, -1])

            tp = np.hstack(image_tp_list)[..., sorted_indices]
            fp = np.hstack(image_fp_list)[..., sorted_indices]

            # calculate recall and precision with tp and fp
            tp_sum = np.cumsum(tp, axis=2)
            fp_sum = np.cumsum(fp, axis=2)
            eps = np.finfo(np.float32).eps
            precisions = tp_sum / np.maximum((tp_sum + fp_sum), eps)
            recalls = tp_sum / np.maximum(num_gts[..., np.newaxis], eps)
            ap = self._calculate_average_precision(precisions, recalls)

            results_per_class.append({
                'num_gts': num_gts,
                'num_dets': len(np.vstack(class_preds)),
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })

        eval_results = self._format_metric_results(results_per_class)
        return eval_results

    def _format_metric_results(self, results_per_class: List[dict]) -> dict:
        """Format the given metric results into a dictionary."""
        eval_results = {}

        for i, iou_thr in enumerate(self.iou_thrs):
            for j, area_range in enumerate(self._area_ranges):
                aps = [res['ap'][i][j] for res in results_per_class]
                if area_range == (None, None):
                    key = f'mAP@{iou_thr}'
                else:
                    key = f'mAP@{iou_thr}@{area_range}'
                eval_results[key] = sum(aps) / len(aps)
        return eval_results
