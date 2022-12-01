import numpy as np
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from .utils.bbox_iou_rotated import bbox_iou_rotated


class DOTAMetric(BaseMetric):
    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence[Dict]):  A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - bboxes (numpy.ndarray): Shape (N, 5), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, 1), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, 1), the predicted labels
                  of bounding boxes.

            groundtruths (Sequence[Dict]):  A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:

                - bboxes (numpy.ndarray): Shape (M, 5), the ground truth
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - labels (numpy.ndarray): Shape (M, 1), the ground truth
                  labels of bounding boxes.
                - bboxes_ignore (numpy.ndarray): Shape (K, 5), the ground
                  truth ignored bounding bboxes of this image,
                  in 'xyxy' foramrt.
                - labels_ignore (numpy.ndarray): Shape (K, 1), the ground
                  truth ignored labels of bounding boxes.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'
            self._results.append((prediction, groundtruth))

    @staticmethod
    def _calculate_image_tpfp(
        pred_bboxes: np.ndarray,
        gt_bboxes: np.ndarray,
        ignore_gt_bboxes: np.ndarray,
        iou_thrs: List[float],
        area_ranges: List[Tuple[Optional[float], Optional[float]]],
        use_legacy_coordinate: bool) -> Tuple[np.ndarray, np.ndarray ]:
        pass
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
                area_mask = filter_by_bboxes_area(pred_bboxes[:, :4], min_area,max_area)
                fp[:, idx, area_mask] = 1
            return tp,fp

        # Step 4. Calculate the IoUs between the predicted bboxes and the
        # ground truth bboxes.
        ious = bbox_iou_rotated(
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
        
    def get_class_predictions(self,
                                predictions: List[dict],
                                class_index: int) -> List:
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

    def calculate_class_tpfp(self,
                            predictions: List[dict],
                            groundtruths: List[dict],
                            class_index: int,
                            pool: Optional[Pool]) -> Tuple:
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
                tpfp = self._calculate_image_tpfp(class_preds       [img_idx],
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
        predictions, groundtruths = zip(*results)
        nproc = min(self.nproc, len(predictions))
        if nproc > 1:
            pool = Pool(nproc)
        else:
            pool = None  # type: ignore