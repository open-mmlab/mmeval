# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics._vendor.scannet import scannet_eval


class InstanceSeg(BaseMetric):
    """3D instance segmentation evaluation metric.

    Args:
        dataset_meta (dict, optional): Provide dataset meta information.
        classes (List[str], optional): Provide dataset classes information as
            an alternative to dataset_meta.
        valid_class_ids (List[int], optional): Provide dataset valid class ids
            information as an alternative to dataset_meta.

    Example:
        >>> import numpy as np
        >>> from mmeval import InstanceSegMetric
        >>> seg_valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24,
        >>>                 28, 33, 34, 36, 39)
        >>> class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
        ...         'window', 'bookshelf', 'picture', 'counter', 'desk',
        ...         'curtain', 'refrigerator', 'showercurtrain', 'toilet',
        ...         'sink', 'bathtub', 'garbagebin')
        >>> dataset_meta = dict(
        ...     seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)
        >>>
        >>> def _demo_mm_model_output(self):
        ...     n_points_list = [3300, 3000]
        ...     gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 1],
        ...                       [13, 13, 2, 1, 3, 3, 0, 0, 0]]
        ...
        ...     predictions = []
        ...     groundtruths = []
        ...
        ...     for n_points, gt_labels in zip(n_points_list, gt_labels_list):
        ...         gt_instance_mask = np.ones(n_points, dtype=int) * -1
        ...         gt_semantic_mask = np.ones(n_points, dtype=int) * -1
        ...         for i, gt_label in enumerate(gt_labels):
        ...             begin = i * 300
        ...             end = begin + 300
        ...             gt_instance_mask[begin:end] = i
        ...             gt_semantic_mask[begin:end] = gt_label
        ...
        ...         ann_info_data = dict()
        ...         ann_info_data['pts_instance_mask'] = torch.tensor(
        ...             gt_instance_mask)
        ...         ann_info_data['pts_semantic_mask'] = torch.tensor(
        ...             gt_semantic_mask)
        ...
        ...         results_dict = dict()
        ...         pred_instance_mask = np.ones(n_points, dtype=int) * -1
        ...         labels = []
        ...         scores = []
        ...         for i, gt_label in enumerate(gt_labels):
        ...             begin = i * 300
        ...             end = begin + 300
        ...             pred_instance_mask[begin:end] = i
        ...             labels.append(gt_label)
        ...             scores.append(.99)
        ...
        ...         results_dict['pts_instance_mask'] = torch.tensor(
        ...             pred_instance_mask)
        ...         results_dict['instance_labels'] = torch.tensor(labels)
        ...         results_dict['instance_scores'] = torch.tensor(scores)
        ...
        ...         predictions.append(results_dict)
        ...         groundtruths.append(ann_info_data)
        ...
        ...     return predictions, groundtruths
        >>>
        >>> instance_seg_metric = InstanceSegMetric(dataset_meta=dataset_meta)
        >>> res = instance_seg_metric(predictions, groundtruths)
        >>> res
        {
        'all_ap': 1.0,
        'all_ap_50%': 1.0,
        'all_ap_25%': 1.0,
        'classes': {
            'cabinet': {
                'ap': 1.0,
                'ap50%': 1.0,
                'ap25%': 1.0
            },
            'bed': {
                'ap': 1.0,
                'ap50%': 1.0,
                'ap25%': 1.0
            },
            'chair': {
                'ap': 1.0,
                'ap50%': 1.0,
                'ap25%': 1.0
            },
            ...
        }
    }
    """

    def __init__(self,
                 classes: Optional[List[str]] = None,
                 valid_class_ids: Optional[List[int]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert (self.dataset_meta
                is not None) or (classes is not None
                                 and valid_class_ids is not None)
        if self.dataset_meta is not None:
            if classes is not None or valid_class_ids is not None:
                warnings.warn('`classes` and `valid_class_ids` are ignored.')
            self.classes = self.dataset_meta['classes']
            self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        else:
            if not isinstance(classes, list):
                raise TypeError('`classes` should be List[str]')
            if not isinstance(valid_class_ids, list):
                raise TypeError('`valid_class_ids` should be List[int]')
            self.classes = classes
            self.valid_class_ids = valid_class_ids

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            predictions (Sequence[Dict]): A sequence of dict. Each dict
                representing a detection result, with the following keys:

                - pts_instance_mask (np.ndarray): Predicted instance masks.
                - instance_labels (np.ndarray): Predicted instance labels.
                - instance_scores (np.ndarray): Predicted instance scores.

            groundtruths (Sequence[Dict]): A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:

                - pts_instance_mask (np.ndarray): Ground truth instance masks.
                - pts_semantic_mask (np.ndarray): Ground truth semantic masks.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            self._results.append((deepcopy(prediction), deepcopy(groundtruth)))

    def compute_metric(self, results: List[List[Dict]]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for result_pred, result_gt in results:
            gt_semantic_masks.append(result_gt['pts_semantic_mask'])
            gt_instance_masks.append(result_gt['pts_instance_mask'])
            pred_instance_masks.append(result_pred['pts_instance_mask'])
            pred_instance_labels.append(result_pred['instance_labels'])
            pred_instance_scores.append(result_pred['instance_scores'])

        ret_dict = self.instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes)

        return ret_dict

    def aggregate_predictions(self, masks, labels, scores, valid_class_ids):
        """Maps predictions to ScanNet evaluator format.

        Args:
            masks (list[np.ndarray]): Per scene predicted instance masks.
            labels (list[np.ndarray]): Per scene predicted instance labels.
            scores (list[np.ndarray]): Per scene predicted instance scores.
            valid_class_ids (tuple[int]): Ids of valid categories.

        Returns:
            list[dict]: Per scene aggregated predictions.
        """
        infos = []
        for id, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            info = dict()
            n_instances = mask.max() + 1
            for i in range(n_instances):
                file_name = f'{id}_{i}'
                info[file_name] = dict()
                info[file_name]['mask'] = (mask == i).astype(int)
                info[file_name]['label_id'] = valid_class_ids[label[i]]
                info[file_name]['conf'] = score[i]
            infos.append(info)
        return infos

    def rename_gt(self, gt_semantic_masks, gt_instance_masks, valid_class_ids):
        """Maps gt instance and semantic masks to instance masks for ScanNet
        evaluator.

        Args:
            gt_semantic_masks (list[np.ndarray]): Per scene gt semantic masks.
            gt_instance_masks (list[np.ndarray]): Per scene gt instance masks.
            valid_class_ids (tuple[int]): Ids of valid categories.

        Returns:
            list[np.array]: Per scene instance masks.
        """
        renamed_instance_masks = []
        for semantic_mask, instance_mask in zip(gt_semantic_masks,
                                                gt_instance_masks):
            unique = np.unique(instance_mask)
            assert len(unique) < 1000
            for i in unique:
                semantic_instance = semantic_mask[instance_mask == i]
                semantic_unique = np.unique(semantic_instance)
                assert len(semantic_unique) == 1
                if semantic_unique[0] < len(valid_class_ids):
                    instance_mask[instance_mask==i] = 1000 * valid_class_ids[semantic_unique[0]] + i  #noqa: E501
            renamed_instance_masks.append(instance_mask)
        return renamed_instance_masks

    def instance_seg_eval(self,
                          gt_semantic_masks,
                          gt_instance_masks,
                          pred_instance_masks,
                          pred_instance_labels,
                          pred_instance_scores,
                          valid_class_ids,
                          class_labels,
                          options=None):
        """Instance Segmentation Evaluation.

        Evaluate the result of the instance segmentation.

        Args:
            gt_semantic_masks (list[np.ndarray]): Ground truth semantic masks.
            gt_instance_masks (list[np.ndarray]): Ground truth instance masks.
            pred_instance_masks (list[np.ndarray]): Predicted instance masks.
            pred_instance_labels (list[np.ndarray]): Predicted instance labels.
            pred_instance_scores (list[np.ndarray]): Predicted instance scores.
            valid_class_ids (tuple[int]): Ids of valid categories.
            class_labels (tuple[str]): Names of valid categories.
            options (dict, optional): Additional options. Keys may contain:
                `overlaps`, `min_region_sizes`, `distance_threshes`,
                `distance_confs`. Default: None.
            logger (logging.Logger | str, optional): The way to print the
                mAP summary. See `mmdet.utils.print_log()` for details.
                Default: None.

        Returns:
            dict[str, float]: Dict of results.
        """
        assert len(valid_class_ids) == len(class_labels)
        id_to_label = {
            valid_class_ids[i]: class_labels[i]
            for i in range(len(valid_class_ids))
        }
        preds = self.aggregate_predictions(
            masks=pred_instance_masks,
            labels=pred_instance_labels,
            scores=pred_instance_scores,
            valid_class_ids=valid_class_ids)
        gts = self.rename_gt(gt_semantic_masks, gt_instance_masks,
                             valid_class_ids)
        metrics = scannet_eval(
            preds=preds,
            gts=gts,
            options=options,
            valid_class_ids=valid_class_ids,
            class_labels=class_labels,
            id_to_label=id_to_label)
        return metrics
