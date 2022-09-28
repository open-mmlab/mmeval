# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from mmeval.detection.utils import (calculate_average_precision,
                                    calculate_overlaps, filter_by_bboxes_area)


class VOCMeanAP(BaseMetric):
    """Pascal VOC evaluation metric.

    This metric compute the VOC mAP (mean Average Precision) with the given IoU
    thresholds and scale ranges.

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
            Defaults to 'area'.
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        classwise_result (bool): Whether to return the computed
            results of each class.
            Defaults to False.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[Tuple]] = None,
                 num_classes: Optional[int] = None,
                 eval_mode: str = 'area',
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 classwise_result: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(iou_thrs, float):
            iou_thrs = [iou_thrs]
        self.iou_thrs = iou_thrs

        if scale_ranges is None:
            scale_ranges = [(None, None)]
        elif (None, None) not in scale_ranges:
            # We allawys compute the mAP across all scale.
            scale_ranges.append((None, None))
        self.scale_ranges = scale_ranges

        area_ranges = []
        for min_scale, max_scale in self.scale_ranges:
            min_area = min_scale if min_scale is None else min_scale**2
            max_area = max_scale if max_scale is None else max_scale**2
            area_ranges.append((min_area, max_area))
        self._area_ranges = area_ranges

        self._num_classes = num_classes

        assert eval_mode in ['area', '11points'], \
            'Unrecognized mode, only "area" and "11points" are supported'
        self.eval_mode = eval_mode

        self.nproc = nproc
        self.use_legacy_coordinate = use_legacy_coordinate
        self.classwise_result = classwise_result

        self.num_iou = len(self.iou_thrs)
        self.num_scale = len(self.scale_ranges)

    @property
    def num_classes(self) -> int:
        """Returns the number of classes.

        The number of classes should be set during initialization, otherwise it
        will be obtained from the 'CLASSES' field in ``self.dataset_meta``.

        Raises:
            RuntimeError: If the num_classes is not set.

        Returns:
            int: The number of classes.
        """
        if self._num_classes is not None:
            return self._num_classes
        if self.dataset_meta and 'CLASSES' in self.dataset_meta:
            self._num_classes = len(self.dataset_meta['CLASSES'])
        else:
            raise RuntimeError(
                "The `num_claases` is required, and also not found 'CLASSES' "
                f'in dataset_meta: {self.dataset_meta}')
        return self._num_classes

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:
                - bboxes, numpy.ndarray with shape (N, 4), the predicted
                bounding bboxes of this image, in 'xyxy' foramrt.
                - scores, numpy.ndarray with shape (N, 1), the predicted scores
                of bounding boxes.
                - labels, numpy.ndarray with shape (N, 1), the predicted labels
                of bounding boxes.
            groundtruths (Sequence[dict]): A sequence of dict. Each dict
                representing a groundtruths for an image, with the following
                keys:
                - bboxes, numpy.ndarray with shape (M, 4), the ground truth
                bounding bboxes of this image, in 'xyxy' foramrt.
                - labels, numpy.ndarray with shape (M, 1), theground truth
                labels of bounding boxes.
                - bboxes_ignore, numpy.ndarray with shape (K, 4), the ground
                truth ignored bounding bboxes of this image,
                in 'xyxy' foramrt.
                - labels_ignore, numpy.ndarray with shape (K, 1), the ground
                truth ignored labels of bounding boxes.
        """
        for prediction, label in zip(predictions, groundtruths):
            self._results.append((prediction, label))

    @staticmethod
    def _calculate_image_tpfp(
            pred_bboxes: np.ndarray, gt_bboxes: np.ndarray,
            ignore_gt_bboxes: np.ndarray, iou_thrs: List[float],
            area_ranges: List[Tuple[Optional[float], Optional[float]]],
            use_legacy_coordinate: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the true positive and false positive on an image.

        Note:
            This method should be a staticmethod to avoid resource competition
            during multiple process.

        Args:
            pred_bboxes (numpy.ndarray): Predicted bboxes of this image, with
                shape (N, 5). The scores The predicted score of the bbox is
                concatenated behind the predicted bbox.
            gt_bboxes (numpy.ndarray): Ground truth bboxes of this image, with
                shape (M, 4).
            ignore_gt_bboxes (numpy.ndarray): Ground truth ignored bboxes of
                this image, with shape (K, 4).
            iou_thrs (List[float]): The IoU thresholds.
            area_ranges (List[Tuple]): The area ranges.
            use_legacy_coordinate (bool): Refer to :class:`VOCMeanAP`.

        Returns:
            tuple (tp, fp):
            - tp, numpy.ndarray with shape (num_ious, num_scales, N),
            the true positive flag of each predicted bbox on this image.
            - fp, numpy.ndarray with shape (num_ious, num_scales, N),
            the false positive flag of each predicted bbox on this image.
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

        # Step 3. If there is no gt bboxes in this image, then all pred bboxes
        # within area range are false positives.
        if all_gt_bboxes.shape[0] == 0:
            for idx, (min_area, max_area) in enumerate(area_ranges):
                area_mask = filter_by_bboxes_area(pred_bboxes[:, :4], min_area,
                                                  max_area)
                fp[:, idx, area_mask] = 1
            return tp, fp

        # Step 4. Calculate the IoUs between the predicted bboxes and the
        # ground truth bboxes.
        ious = calculate_overlaps(
            pred_bboxes[:, :4],
            all_gt_bboxes,
            mode='iou',
            use_legacy_coordinate=use_legacy_coordinate)
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
                gt_area_mask = filter_by_bboxes_area(all_gt_bboxes, min_area,
                                                     max_area)
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
                        area_mask = filter_by_bboxes_area(
                            pred_bboxes[pred_bbox_idx, :4], min_area, max_area)
                        if area_mask:
                            fp[iou_thr_idx, area_idx, pred_bbox_idx] = 1

        return tp, fp

    def get_class_predictions(self, predictions: List[dict],
                              class_index: int) -> List:
        """Get prediciton results of a certain class index.

        Args:
            predictions (list[dict]): Same as :class:`VOCMeanAP.add`.
            class_index (int): Index of a specific class.

        Returns:
            list[np.ndarray]: A list of predicted bboxes of this class. Each
            predicted score of the bbox is concatenated behind the predicted
            bbox.
        """
        class_preds = []
        for pred in predictions:
            pred_indices = (pred['labels'] == class_index)
            pred_bboxes_info = np.concatenate(
                (pred['bboxes'][pred_indices, :],
                 pred['scores'][pred_indices].reshape((-1, 1))),
                axis=1)
            class_preds.append(pred_bboxes_info)
        return class_preds

    def get_class_gts(self, groundtruths: List[dict],
                      class_index: int) -> Tuple:
        """Get prediciton gt information of a certain class index.

        Args:
            groundtruths (list[dict]): Same as :class:`VOCMeanAP.add`.
            class_index (int): Index of a specific class.

        Returns:
            tuple (class_gts, class_ignore_gts):
            - class_gts, List[numpy.ndarray], The gt bboxes of this class.
            - class_ignore_gts, List[numpy.ndarray], The ignored gt bboxes of
            this class. This is necessary when counting tp and fp.
        """
        class_gts = []
        class_ignore_gts = []

        for gt in groundtruths:
            gt_indices = (gt['labels'] == class_index)
            gt_bboxes = gt['bboxes'][gt_indices, :]
            ignore_gt_indices = (gt['labels_ignore'] == class_index)
            ignore_gt_bboxes = gt['bboxes_ignore'][ignore_gt_indices, :]
            class_gts.append(gt_bboxes)
            class_ignore_gts.append(ignore_gt_bboxes)

        return class_gts, class_ignore_gts

    def calculate_class_tpfp(self, predictions: List[dict],
                             groundtruths: List[dict], class_index: int,
                             pool: Optional[Pool]) -> Tuple:
        """Calculate the tp and fp of the given class index.

        Args:
            predictions (List[dict]): A list of dict. Each dict is the
                detection result of an image. Same as :class:`VOCMeanAP.add`.
            groundtruths (List[dict]): A list of dict. Each dict is the
                ground truth of an image. Same as :class:`VOCMeanAP.add`.
            class_index (int): The class index.
            pool (Optional[Pool]): A instance of :class:`multiprocessing.Pool`.
                If None, do not use multiprocessing.

        Returns:
            tuple (tp, fp, num_gts):
            - tp, numpy.ndarray with shape (num_ious, num_scales, num_pred),
            the true positive flag of each predicted bbox for this class.
            - fp, numpy.ndarray with shape (num_ious, num_scales, num_pred),
            the false positive flag of each predicted bbox for this class.
            - num_gts, numpy.ndarray with shape (num_ious, num_scales), the
            number of ground truths.
        """
        class_preds = self.get_class_predictions(predictions, class_index)
        class_gts, class_ignore_gts = self.get_class_gts(
            groundtruths, class_index)

        if pool is not None:
            num_images = len(class_preds)
            tpfp_list = pool.starmap(
                self._calculate_image_tpfp,
                zip(class_preds, class_gts, class_ignore_gts,
                    [self.iou_thrs] * num_images,
                    [self._area_ranges] * num_images,
                    [self.use_legacy_coordinate] * num_images))
        else:
            tpfp_list = []
            for img_idx in range(len(class_preds)):
                tpfp = self._calculate_image_tpfp(class_preds[img_idx],
                                                  class_gts[img_idx],
                                                  class_ignore_gts[img_idx],
                                                  self.iou_thrs,
                                                  self._area_ranges,
                                                  self.use_legacy_coordinate)
                tpfp_list.append(tpfp)

        image_tp_list, image_fp_list = tuple(zip(*tpfp_list))
        sorted_indices = np.argsort(-np.vstack(class_preds)[:, -1])
        tp = np.concatenate(image_tp_list, axis=2)[..., sorted_indices]
        fp = np.concatenate(image_fp_list, axis=2)[..., sorted_indices]

        num_gts = np.zeros((self.num_iou, self.num_scale), dtype=int)
        for idx, (min_area, max_area) in enumerate(self._area_ranges):
            area_mask = filter_by_bboxes_area(
                np.vstack(class_gts), min_area, max_area)
            num_gts[:, idx] = np.sum(area_mask)

        return tp, fp, num_gts

    def compute_metric(self, results: list) -> dict:
        """Compute the VOCMeanAP metric.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric, with the following keys:
            - mAP, the averaged across all IoU thresholds and all class.
            - mAP@{IoU}, the mAP of the specified IoU threshold.
            - mAP@{scale_range}, the mAP of the specified scale range.
            - classwise_result, the evaluate results of each classes.
            This would be returned if ``self.classwise_result`` is True.
        """
        predictions, groundtruths = zip(*results)

        nproc = min(self.nproc, len(predictions))
        if nproc > 1:
            pool = Pool(nproc)
        else:
            pool = None  # type: ignore

        results_per_class = []
        for class_index in range(self.num_classes):
            # Calculate tp, fp and num_gts.
            tp, fp, num_gts = self.calculate_class_tpfp(
                predictions, groundtruths, class_index, pool)

            # Calculate recalls and precisions.
            tp_cumsum = np.cumsum(tp, axis=2)
            fp_cumsum = np.cumsum(fp, axis=2)
            eps = np.finfo(np.float32).eps
            precisions = tp_cumsum / np.maximum((tp_cumsum + fp_cumsum), eps)
            recalls = tp_cumsum / np.maximum(num_gts[..., np.newaxis], eps)

            # Calculate average precision per `iou_thr` and `scale_range`.
            ap = np.zeros((self.num_iou, self.num_scale), dtype=np.float32)
            for i in range(self.num_iou):
                for j in range(self.num_scale):
                    ap[i, j] = calculate_average_precision(
                        recalls[i, j], precisions[i, j], self.eval_mode)

            results_per_class.append({
                'num_gts': num_gts,
                'num_dets': tp.shape[-1],
                'recalls': recalls,
                'precisions': precisions,
                'ap': ap,
            })

        if pool is not None:
            pool.close()

        eval_results = self._aggregate_results(results_per_class)
        if self.classwise_result:
            eval_results['classwise_result'] = results_per_class

        return eval_results

    def _aggregate_results(self, results_per_class: List[dict]) -> dict:
        """Aggregate class-wise results and return a dictionary.

        Args:
            results_per_class (List[dict]): The class-wise evaluate results.

        Returns:
            dict: The aggregated metric results, with the following keys:
            - mAP, the averaged across all IoU thresholds and all class.
            - mAP@{IoU}, the mAP of the specified IoU threshold.
            - mAP@{scale_range}, the mAP of the specified scale range.
        """
        eval_results = {}

        for i, iou_thr in enumerate(self.iou_thrs):
            for j, scale_range in enumerate(self.scale_ranges):
                if scale_range != (None, None):
                    continue
                aps = [
                    res['ap'][i][j] for res in results_per_class
                    if res['num_gts'][i][j] > 0
                ]
                eval_results[f'mAP@{iou_thr}'] = np.array(aps).mean().item()

        for j, scale_range in enumerate(self.scale_ranges):
            ap_per_ious = []
            for i in range(len(self.iou_thrs)):
                aps = [
                    res['ap'][i][j] for res in results_per_class
                    if res['num_gts'][i][j] > 0
                ]
                ap_per_ious.append(np.array(aps).mean().item())
            if scale_range == (None, None):
                key = 'mAP'
            else:
                key = f'mAP@{scale_range}'
            eval_results[key] = np.array(ap_per_ious).mean().item()

        return eval_results
