# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations
import copy
from typing import List, Optional, Tuple, Union
import numpy as np

from mmeval.detection.voc_map import VOCMeanAP
from mmeval.detection.utils import (calculate_overlaps,
                                    filter_by_bboxes_area)


class OIDMeanAP(VOCMeanAP):

    def __init__(self,
                 iof_thrs: Union[float, List[float]] = 0.5,
                 use_group_of: bool = True,
                 get_supercategory: bool = True,
                 filter_labels: bool = True,
                 class_relation_matrix: Optional[np.ndarray] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(iof_thrs, float):
            iof_thrs = (iof_thrs, )
        self.iof_thrs = iof_thrs
        self.num_iof = len(iof_thrs)

        self.use_group_of = use_group_of
        self.get_supercategory = get_supercategory
        self.filter_labels = filter_labels
        self._class_relation_matrix = class_relation_matrix

    @property
    def class_relation_matrix(self) -> int:
        """Returns the class relation matrix.

        The class relation matrix should be set during initialization,
        otherwise it will be obtained from the 'RELATION_MATRIX' field in
        ``self.dataset_meta``.

        Raises:
            RuntimeError: If the num_classes is not set.

        Returns:
            int: The number of classes.
        """
        if self._class_relation_matrix is not None:
            return self._class_relation_matrix
        if self.dataset_meta and 'RELATION_MATRIX' in self.dataset_meta:
            self._class_relation_matrix = self.dataset_meta['RELATION_MATRIX']
        else:
            raise RuntimeError(
                "The `class_relation_matrix` is required, and also not found"
                f" 'RELATION_MATRIX' in dataset_meta: {self.dataset_meta}")
        return self._class_relation_matrix

    def _expand_supercategory_ann(self, instances: List[dict]) -> List[dict]:
        """Get parent classes's annotation of the corresponding class.

        Args:
            instances (List[dict]): A list of annotations of the instances.

        Returns:
            List[dict]: Annotations extended with super-category.
        """
        supercat_instances = []
        relation_matrix = self.class_relation_matrix
        for instance in instances:
            # Find the super-category by class relation matrix.
            supercats = np.where(relation_matrix[instance['bbox_label']])[0]
            for supercat in supercats:
                if supercat == instance['bbox_label']:
                    continue
                # Copy this instance and change the 'bbox_label'
                new_instance = copy.deepcopy(instance)
                new_instance['bbox_label'] = supercat
                supercat_instances.append(new_instance)
        return supercat_instances

    def _process_prediction(self, pred, allowed_labels):
        origin_pred = copy.deepcopy(pred)
        relation_matrix = self.class_relation_matrix
        for pred_label in np.unique(pred['labels']):
            supercat_labels = np.where(relation_matrix[pred_label])[0]
            for supercat in supercat_labels:
                if (supercat in allowed_labels and supercat != pred_label
                        and self.get_supercategory):
                    # add super-supercategory preds
                    indices = np.where(origin_pred['labels'] == pred_label)[0]
                    pred['scores'] = np.concatenate(
                        [pred['scores'], origin_pred['scores'][indices]])
                    pred['bboxes'] = np.concatenate(
                        [pred['bboxes'], origin_pred['bboxes'][indices]])
                    extend_labels = np.full(indices.shape, supercat, dtype=np.int64)
                    pred['labels'] = np.concatenate([pred['labels'], extend_labels])
                elif supercat not in allowed_labels and self.filter_labels:
                    indices = np.where(pred['labels'] != supercat)[0]
                    pred['scores'] = pred['scores'][indices]
                    pred['bboxes'] = pred['bboxes'][indices]
                    pred['labels'] = pred['labels'][indices]
        return pred

    def _stack_gt_instances(self, instances):
        gt_labels, gt_bboxes, is_group_ofs = [], [], []
        for ins in instances:
            gt_labels.append(ins['bbox_label'])
            gt_bboxes.append(ins['bbox'])
            is_group_ofs.append(ins['is_group_of'])
        annotation = {
            "bboxes": np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4)),
            "labels": np.array(gt_labels, dtype=np.int64),
            "is_group_ofs": np.array(is_group_ofs, dtype=bool),
        }
        return annotation

    def add(self, predictions, groundtruths):
        for pred, groundtruth in zip(predictions, groundtruths):
            instances = copy.deepcopy(groundtruth['instances'])
            pred = copy.deepcopy(pred)
            if self.get_supercategory:
                supercat_instances = self._expand_supercategory_ann(instances)
                instances.extend(supercat_instances)
            gt_ann = self._stack_gt_instances(instances)

            if 'image_level_labels' not in groundtruth:
                allowed_labels = np.unique(gt_ann['labels'])
            else:
                allowed_labels = np.unique(np.append(gt_ann['labels'],
                    groundtruth['image_level_labels']))
            pred = self._process_prediction(pred, allowed_labels)

            self._results.append((pred, gt_ann))

    @staticmethod
    def _calculate_image_tpfp(
            pred_bboxes: np.ndarray,
            gt_bboxes: np.ndarray,
            group_of_flags: np.ndarray,
            use_group_of: bool,
            iof_thrs: List[float],
            iou_thrs: List[float],
            area_ranges: List[Tuple[Optional[float], Optional[float]]],
            use_legacy_coordinate: bool) -> Tuple[np.ndarray, np.ndarray]:
        
        ignore_gt_bboxes = np.empty((0, 4))

        if not use_group_of:
            tp, fp = VOCMeanAP._calculate_image_tpfp(
                pred_bboxes,
                gt_bboxes,
                ignore_gt_bboxes,
                iou_thrs,
                area_ranges,
                use_legacy_coordinate)
            return tp, fp, pred_bboxes

        non_group_gt_bboxes = gt_bboxes[~group_of_flags]
        group_gt_bboxes = gt_bboxes[group_of_flags]
        tp, fp = VOCMeanAP._calculate_image_tpfp(
            pred_bboxes,
            non_group_gt_bboxes,
            ignore_gt_bboxes,
            iou_thrs,
            area_ranges,
            use_legacy_coordinate)
        
        if group_gt_bboxes.shape[0] == 0:
            return tp, fp, pred_bboxes
        
        iofs = calculate_overlaps(
            pred_bboxes[:, :4],
            group_gt_bboxes,
            mode='iof',
            use_legacy_coordinate=use_legacy_coordinate)
        
        num_iou = len(iou_thrs)
        num_scale = len(area_ranges)
        pred_bboxes_group = np.zeros(
            (num_iou, num_scale,
             iofs.shape[1], pred_bboxes.shape[1]), dtype=float)
        match_group_of = np.zeros(
            (num_iou, num_scale, pred_bboxes.shape[0]), dtype=bool)
        tp_group = np.zeros(
            (num_iou, num_scale, group_gt_bboxes.shape[0]), dtype=np.float32)

        iofs_max = iofs.max(axis=1)
        # for each det, which gt overlaps most with it
        iofs_argmax = iofs.argmax(axis=1)
        # sort all dets in descending order by scores
        sorted_indices = np.argsort(-pred_bboxes[:, -1])

        for iof_thr_idx, iof_thr in enumerate(iof_thrs):
            for area_idx, (min_area, max_area) in enumerate(area_ranges):
                box_is_covered = tp[iof_thr_idx, area_idx]
                # if no area range is specified, gt_area_ignore is all False
                gt_area_mask = filter_by_bboxes_area(
                    gt_bboxes, min_area, max_area)
                ignore_gt_area_flags = ~gt_area_mask

                for pred_bbox_idx in sorted_indices:
                    if box_is_covered[pred_bbox_idx]:
                        continue
                    if iofs_max[pred_bbox_idx] < iof_thr:
                        continue

                    matched_gt_idx = iofs_argmax[pred_bbox_idx]
                    if ignore_gt_area_flags[matched_gt_idx]:
                        continue

                    if not tp_group[iof_thr_idx, area_idx , matched_gt_idx]:
                        tp_group[iof_thr_idx, area_idx, matched_gt_idx] = 1
                        match_group_of[iof_thr_idx, area_idx, pred_bbox_idx] = True
                    else:
                        match_group_of[iof_thr_idx, area_idx, pred_bbox_idx] = True
                    
                    if pred_bboxes_group[iof_thr_idx, area_idx, matched_gt_idx, -1] < pred_bboxes[pred_bbox_idx, -1]:
                        pred_bboxes_group[iof_thr_idx, area_idx, matched_gt_idx] = pred_bboxes[pred_bbox_idx]

        fp_group = (tp_group <= 0).astype(float)
        tps_list = []
        fps_list = []
        # concatenate tp, fp, and det-boxes which not matched group of
        # gt boxes and tp_group, fp_group, and pred_bboxes_group which
        # matched group of boxes respectively.
        for i in range(num_iou):
            tps, fps = [], []
            for j in range(num_scale):
                tps.append(
                    np.concatenate((tp[i, j][~match_group_of[i, j]], tp_group[i, j])))
                fps.append(
                    np.concatenate((fp[i, j][~match_group_of[i, j]], fp_group[i, j])))
                pred_bboxes = np.concatenate(
                    (pred_bboxes[~match_group_of[i, j]], pred_bboxes_group[i, j]))
            tps_list.append(np.vstack(tps))
            fps_list.append(np.vstack(fps))

        tp = np.vstack(tps_list).reshape((num_iou, num_scale, -1))
        fp = np.vstack(fps_list).reshape((num_iou, num_scale, -1))
        return tp, fp, pred_bboxes

    def get_class_gts(self, groundtruths: List[dict], class_index: int) -> Tuple:
        """Get prediciton gt information of a certain class index.

        Args:
            groundtruths (list[dict]): Same as ``self.add``.
            class_index (int): Index of a specific class.

        Returns:
            list[np.ndarray]: gt bboxes.
        """
        class_gts = []
        class_group_of_flags = []

        for gt in groundtruths:
            gt_indices = (gt['labels'] == class_index)
            gt_bboxes = gt['bboxes'][gt_indices, :]
            group_of_flags = gt['is_group_ofs'][gt_indices]
            class_gts.append(gt_bboxes)
            class_group_of_flags.append(group_of_flags)

        return class_gts, class_group_of_flags
    
    def calculate_class_tpfp(self, predictions, groundtruths, class_index, pool):
        class_preds = self.get_class_predictions(predictions, class_index)
        class_gts, class_group_of_flags = self.get_class_gts(
            groundtruths, class_index)

        if pool is not None:
            num_images = len(class_preds)
            tpfp_list = pool.starmap(
                self._calculate_image_tpfp,
                zip(class_preds,
                    class_gts,
                    class_group_of_flags,
                    [self.use_group_of, ] * num_images,
                    [self.iof_thrs, ] * num_images,
                    [self.iou_thrs, ] * num_images,
                    [self._area_ranges, ] * num_images,
                    [self.use_legacy_coordinate, ] * num_images))
        else:
            tpfp_list = []
            for img_idx in range(len(class_preds)):
                tp, fp, img_pred  = self._calculate_image_tpfp(
                    class_preds[img_idx],
                    class_gts[img_idx],
                    class_group_of_flags[img_idx],
                    self.use_group_of,
                    self.iof_thrs,
                    self.iou_thrs,
                    self._area_ranges,
                    self.use_legacy_coordinate)
                tpfp_list.append((tp, fp, img_pred))

        image_tp_list, image_fp_list, class_preds = tuple(zip(*tpfp_list))
        sorted_indices = np.argsort(-np.vstack(class_preds)[:, -1])
        tp = np.concatenate(image_tp_list, axis=2)[..., sorted_indices]
        fp = np.concatenate(image_fp_list, axis=2)[..., sorted_indices]

        num_gts = np.zeros((self.num_iou, self.num_scale), dtype=int)
        for idx, (min_area, max_area) in enumerate(self._area_ranges):
            area_mask = filter_by_bboxes_area(
                np.vstack(class_gts), min_area, max_area)
            num_gts[:, idx] = np.sum(area_mask)

        return tp, fp, num_gts
