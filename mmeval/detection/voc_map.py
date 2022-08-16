# Copyright (c) OpenMMLab. All rights reserved.

import multiprocessing
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from .utils.bbox import calculate_bboxes_area, calculate_overlaps


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
                 scale_ranges: Optional[List[tuple]] = None,
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

    def _calculate_tpfp(
            self, pred_bboxes: np.ndarray, gt_bboxes: np.ndarray,
            ignore_gt_bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Check if detected bboxes are true positive or false positive.

        Args:
            pred_bboxes (numpy.ndarray): Detected bboxes of this image, of
                shape (m, 5).
            gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
            gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
                of shape (k, 4). Defaults to None

        Returns:
            tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape
            of each array is (num_ious, num_scales, m).
        """
        # an indicator of ignored gts
        gt_ignore_indices = np.concatenate(
            (np.zeros(gt_bboxes.shape[0], dtype=bool),
             np.ones(ignore_gt_bboxes.shape[0], dtype=bool)))
        # stack gt_bboxes and gt_bboxes_ignore for convenience
        gt_bboxes = np.vstack((gt_bboxes, ignore_gt_bboxes))

        num_preds = pred_bboxes.shape[0]
        num_gts = gt_bboxes.shape[0]
        num_iou_thrs = len(self.iou_thrs)
        num_scales = len(self.scale_ranges)

        tp = np.zeros((num_iou_thrs, num_scales, num_preds), dtype=np.float32)
        fp = np.zeros((num_iou_thrs, num_scales, num_preds), dtype=np.float32)

        # if there is no gt bboxes in this image, then all det bboxes
        # within area range are false positives
        if num_gts == 0:
            if self._area_ranges == [(None, None)]:
                fp[...] = 1
            else:
                pred_areas = calculate_bboxes_area(pred_bboxes[..., :4])
                for scale_idx, (min_area,
                                max_area) in enumerate(self._area_ranges):
                    mask = (pred_areas >= min_area) & (pred_areas < max_area)
                    fp[:, scale_idx, mask] = 1
            return tp, fp

        ious = calculate_overlaps(
            pred_bboxes,
            gt_bboxes,
            mode='iou',
            use_legacy_coordinate=self.use_legacy_coordinate)
        # for each det, the max iou with all gts
        ious_max = ious.max(axis=1)
        # for each det, which gt overlaps most with it
        ious_argmax = ious.argmax(axis=1)
        # sort all dets in descending order by scores
        sorted_indices = np.argsort(-pred_bboxes[:, -1])

        for iou_thr_idx, iou_thr in enumerate(self.iou_thrs):
            for area_idx, (min_area, max_area) in enumerate(self._area_ranges):
                gt_covered = np.zeros(num_gts, dtype=bool)
                if min_area is None:
                    gt_area_ignore = np.zeros_like(
                        gt_ignore_indices, dtype=bool)
                else:
                    gt_areas = calculate_bboxes_area(
                        gt_bboxes, self.use_legacy_coordinate)
                    gt_area_ignore = (gt_areas < min_area) | (
                        gt_areas >= max_area)

                for pred_bbox_idx in sorted_indices:
                    if ious_max[pred_bbox_idx] >= iou_thr:
                        matched_gt = ious_argmax[pred_bbox_idx]
                        if not (gt_ignore_indices[matched_gt]
                                or gt_area_ignore[matched_gt]):
                            if not gt_covered[matched_gt]:
                                gt_covered[matched_gt] = True
                                tp[iou_thr_idx, area_idx, pred_bbox_idx] = 1
                            else:
                                fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1
                        # otherwise ignore this detected bbox, tp = 0, fp = 0
                    elif min_area is None:
                        fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1
                    else:
                        area = calculate_bboxes_area(
                            pred_bboxes[pred_bbox_idx, :4],
                            self.use_legacy_coordinate)
                        if area >= min_area and area < max_area:
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
            tuple[list[np.ndarray]]: predictions bboxes, gt bboxes,
                ignored gt bboxes.
        """
        class_preds = []
        class_gts = []
        class_ignore_gts = []

        for pred, gt in zip(predictions, groundtruths):
            pred_indices = (pred['labels'] == class_index)
            bboxes_with_scores = np.hstack([
                pred['bboxes'][pred_indices, :],
                pred['scores'][pred_indices].reshape((-1, 1))
            ])
            class_preds.append(bboxes_with_scores)

            gt_indices = (gt['labels'] == class_index)
            class_gts.append(gt['bboxes'][gt_indices, :])

            if gt.get('labels_ignore', None) is not None:
                ignore_gt_indices = (gt['labels_ignore'] == class_index)
                class_ignore_gts.append(
                    gt['bboxes_ignore'][ignore_gt_indices, :])
            else:
                class_ignore_gts.append(np.empty((0, 4), dtype=np.float32))
        return class_preds, class_gts, class_ignore_gts

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
            class_preds, class_gts, class_ignore_gts = self.get_class_results(
                predictions, groundtruths, class_index)

            if pool is not None:
                tpfp_list = pool.starmap(
                    self._calculate_tpfp,
                    zip(class_preds, class_gts, class_ignore_gts))
            else:
                tpfp_list = [
                    self._calculate_tpfp(class_preds[i], class_gts[i],
                                         class_ignore_gts[i])
                    for i in range(num_images)
                ]

            image_tp_list, image_fp_list = tuple(zip(*tpfp_list))

            num_gts = np.zeros((len(self.iou_thrs), len(self.scale_ranges)),
                               dtype=int)

            for image_idx in range(num_images):
                if self._area_ranges == [(None, None)]:
                    num_gts[:, 0] += class_gts[image_idx].shape[0]
                else:
                    gt_areas = calculate_bboxes_area(
                        class_gts[image_idx], self.use_legacy_coordinate)
                    for k, (min_area,
                            max_area) in enumerate(self._area_ranges):
                        num_gts[:, k] += np.sum((gt_areas >= min_area)
                                                & (gt_areas < max_area))

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
            if self.scale_ranges == [
                (None, None),
            ]:
                aps = [res['ap'][i][0] for res in results_per_class]
                eval_results[f'mAP@{iou_thr}'] = sum(aps) / len(aps)
            else:
                for j, scale in enumerate(self.scale_ranges):
                    aps = [res['ap'][i][j] for res in results_per_class]
                    eval_results[f'mAP@{iou_thr}@{scale}'] = sum(aps) / len(
                        aps)

        return eval_results
