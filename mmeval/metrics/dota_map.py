# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Sequence, Tuple

from mmeval.metrics.voc_map import VOCMeanAP
from .utils.bbox_overlaps_rotated import calculate_bboxes_area_rotated

logger = logging.getLogger(__name__)

try:
    # we prefer to use `bbox_iou_rotated` in mmcv
    from mmcv.ops import box_iou_rotated as calculate_overlaps_rotated
except Exception as e:  # noqa F841
    from .utils.bbox_overlaps_rotated import calculate_overlaps_rotated
    logger.debug(
        'mmcv is not installed, calculating IoU of rotated bbox with OpenCV.')


def filter_by_bboxes_area_rotated(bboxes: np.ndarray,
                                  min_area: Optional[float],
                                  max_area: Optional[float]):
    """Filter the rotated bboxes with an area range.

    Args:
        bboxes (numpy.ndarray): The bboxes with shape (n, 5) in 'xywha' format.
        min_area (Optional[float]): The minimum area. If None, does not filter
            the minimum area.
        max_area (Optional[float]): The maximum area. If None, does not filter
            the maximum area.

    Returns:
        numpy.ndarray: A mask of ``bboxes`` identify which bbox are filtered.
    """
    bboxes_area = calculate_bboxes_area_rotated(bboxes)
    area_mask = np.ones_like(bboxes_area, dtype=bool)
    if min_area is not None:
        area_mask &= (bboxes_area >= min_area)
    if max_area is not None:
        area_mask &= (bboxes_area < max_area)
    return area_mask


class DOTAMeanAP(VOCMeanAP):
    """DOTA evaluation metric.

    DOTA is a large-scale dataset for object detection in aerial images which
    is introduced in https://arxiv.org/abs/1711.10398. This metric computes
    the DOTA mAP (mean Average Precision) with the given IoU thresholds and
    scale ranges.

    Args:
        iou_thrs (float ｜ List[float]): IoU thresholds. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        num_classes (int, optional): The number of classes. If None, it will be
            obtained from the 'CLASSES' field in ``self.dataset_meta``.
            Defaults to None.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'.
            Defaults to '11points'.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        drop_class_ap (bool): Whether to drop the class without ground truth
            when calculating the average precision for each class.
        classwise (bool): Whether to return the computed results of each
            class. Defaults to False.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> import numpy as np
        >>> from mmeval import DOTAMetric
        >>> num_classes = 15
        >>> dota_metric = DOTAMetric(num_classes=15)
        >>>
        >>> def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
        ...     # random generate bounding boxes in 'xywha' formart.
        ...     x = np.random.rand(num_bboxes, ) * img_w
        ...     y = np.random.rand(num_bboxes, ) * img_h
        ...     w = np.random.rand(num_bboxes, ) * (img_w - x)
        ...     h = np.random.rand(num_bboxes, ) * (img_h - y)
        ...     a = np.random.rand(num_bboxes, ) * np.pi / 2
        ...     return np.stack([x, y, w, h, a], axis=1)
        >>> prediction = {
        ...     'bboxes': _gen_bboxes(10),
        ...     'scores': np.random.rand(10, ),
        ...     'labels': np.random.randint(0, num_classes, size=(10, ))
        ... }
        >>> groundtruth = {
        ...     'bboxes': _gen_bboxes(10),
        ...     'labels': np.random.randint(0, num_classes, size=(10, )),
        ...     'bboxes_ignore': _gen_bboxes(5),
        ...     'labels_ignore': np.random.randint(0, num_classes, size=(5, ))
        ... }
        >>> dota_metric(predictions=[prediction, ], groundtruths=[groundtruth, ])  # doctest: +ELLIPSIS  # noqa: E501
        {'mAP@0.5': ..., 'mAP': ...}
    """

    def __init__(self, eval_mode: str = '11points', **kwargs) -> None:
        super().__init__(eval_mode=eval_mode, **kwargs)

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence[Dict]):  A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:
                - bboxes (numpy.ndarray): Shape (N, 5), the predicted
                  bounding bboxes of this image, in 'xywha' foramrt.
                - scores (numpy.ndarray): Shape (N, ), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
            groundtruths (Sequence[Dict]):  A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:
                - bboxes (numpy.ndarray): Shape (M, 5), the ground truth
                  bounding bboxes of this image, in 'xywha' foramrt.
                - labels (numpy.ndarray): Shape (M, ), the ground truth
                  labels of bounding boxes.
                - bboxes_ignore (numpy.ndarray): Shape (K, 5), the ground
                  truth ignored bounding bboxes of this image,
                  in 'xywha' foramrt.
                - labels_ignore (numpy.ndarray): Shape (K, ), the ground
                  truth ignored labels of bounding boxes.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'  # noqa: E501
            self._results.append((prediction, groundtruth))

    @staticmethod
    def _calculate_image_tpfp(  # type: ignore
        pred_bboxes: np.ndarray,
        gt_bboxes: np.ndarray,
        ignore_gt_bboxes: np.ndarray,
        iou_thrs: List[float],
        area_ranges: List[Tuple[Optional[float], Optional[float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the true positive and false positive on an image.

        Args:
            pred_bboxes (numpy.ndarray): Predicted bboxes of this image, with
                shape (N, 6). The scores The predicted score of the bbox is
                concatenated behind the predicted bbox.
            gt_bboxes (numpy.ndarray): Ground truth bboxes of this image, with
                shape (M, 5).
            ignore_gt_bboxes (numpy.ndarray): Ground truth ignored bboxes of
                this image, with shape (K, 5).
            iou_thrs (List[float]): The IoU thresholds.
            area_ranges (List[Tuple]): The area ranges.

        Returns:
            tuple (tp, fp):
            - tp (numpy.ndarray): Shape (num_ious, num_scales, N),
              the true positive flag of each predicted bbox on this image.
            - fp (numpy.ndarray): Shape (num_ious, num_scales, N),
              the false positive flag of each predicted bbox on this image.

        Note:
            This method should be a staticmethod to avoid resource competition
            during multiple processes.
        """
        # Step 1. Concatenate `gt_bboxes` and `ignore_gt_bboxes`, then set
        # the `ignore_gt_flags`.
        all_gt_bboxes = np.concatenate((gt_bboxes, ignore_gt_bboxes))
        ignore_gt_flags = np.concatenate((np.zeros(
            (gt_bboxes.shape[0], 1),
            dtype=bool), np.ones((ignore_gt_bboxes.shape[0], 1), dtype=bool)))

        # Step 2. Initialize the `tp` and `fp` arrays.
        num_preds = pred_bboxes.shape[0]
        tp = np.zeros((len(iou_thrs), len(area_ranges), num_preds))
        fp = np.zeros((len(iou_thrs), len(area_ranges), num_preds))

        # Step 3. If there are no gt bboxes in this image, then all pred bboxes
        # within area range are false positives.
        if all_gt_bboxes.shape[0] == 0:
            for idx, (min_area, max_area) in enumerate(area_ranges):
                area_mask = filter_by_bboxes_area_rotated(
                    pred_bboxes[:, :5], min_area, max_area)
                fp[:, idx, area_mask] = 1
            return tp, fp

        # Step 4. Calculate the IoUs between the predicted bboxes and the
        # ground truth bboxes.
        ious = calculate_overlaps_rotated(pred_bboxes[:, :5], all_gt_bboxes)
        # For each pred bbox, the max iou with all gts.
        ious_max = ious.max(axis=1)
        # For each pred bbox, which gt overlaps most with it.
        ious_argmax = ious.argmax(axis=1)
        # Sort all pred bbox in descending order by scores.
        sorted_indices = np.argsort(-pred_bboxes[:, -1])

        # Step 5. Count the `tp` and `fp` of each iou threshold and area range.
        for iou_thr_idx, iou_thr in enumerate(iou_thrs):
            for area_idx, (min_area, max_area) in enumerate(area_ranges):
                # The flags that gt bboxes have been matched.
                gt_covered_flags = np.zeros(all_gt_bboxes.shape[0], dtype=bool)
                # The flags that gt bboxes out of area range.
                gt_area_mask = filter_by_bboxes_area_rotated(
                    all_gt_bboxes, min_area, max_area)
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
                        area_mask = filter_by_bboxes_area_rotated(
                            pred_bboxes[pred_bbox_idx, :5], min_area, max_area)
                        if area_mask:
                            fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1

        return tp, fp

    def calculate_class_tpfp(self, predictions: List[dict],
                             groundtruths: List[dict], class_index: int,
                             pool: Optional[Pool]) -> Tuple:
        """Calculate the tp and fp of the given class index.

        Args:
            predictions (List[dict]): A list of dict. Each dict is the
                detection result of an image.
            groundtruths (List[dict]): A list of dict. Each dict is the
                ground truth of an image.
            class_index (int): The class index.
            pool (Optional[Pool]): A instance of :class:`multiprocessing.Pool`.
                If None, do not use multiprocessing.

        Returns:
            tuple (tp, fp, num_gts):
            - tp (numpy.ndarray): Shape (num_ious, num_scales, num_pred),
              the true positive flag of each predicted bbox for this class.
            - fp (numpy.ndarray): Shape (num_ious, num_scales, num_pred),
              the false positive flag of each predicted bbox for this class.
            - num_gts (numpy.ndarray): Shape (num_ious, num_scales), the
              number of ground truths.
        """
        class_preds = self.get_class_predictions(predictions, class_index)
        class_gts, class_ignore_gts = self.get_class_gts(
            groundtruths, class_index)
        if pool is not None:
            num_images = len(class_preds)
            tpfp_list = pool.starmap(
                self._calculate_image_tpfp,
                zip(
                    class_preds,
                    class_gts,
                    class_ignore_gts,
                    [self.iou_thrs] * num_images,
                    [self._area_ranges] * num_images,
                ))
        else:
            tpfp_list = []
            for img_idx in range(len(class_preds)):
                tpfp = self._calculate_image_tpfp(class_preds[img_idx],
                                                  class_gts[img_idx],
                                                  class_ignore_gts[img_idx],
                                                  self.iou_thrs,
                                                  self._area_ranges)
                tpfp_list.append(tpfp)

        image_tp_list, image_fp_list = tuple(zip(*tpfp_list))
        sorted_indices = np.argsort(-np.vstack(class_preds)[:, -1])
        tp = np.concatenate(image_tp_list, axis=2)[..., sorted_indices]
        fp = np.concatenate(image_fp_list, axis=2)[..., sorted_indices]
        num_gts = np.zeros((self.num_iou, self.num_scale), dtype=int)
        for idx, (min_area, max_area) in enumerate(self._area_ranges):
            area_mask = filter_by_bboxes_area_rotated(
                np.vstack(class_gts), min_area, max_area)
            num_gts[:, idx] = np.sum(area_mask)

        return tp, fp, num_gts
