# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics._vendor.scannet import scannet_eval


class InstanceSeg(BaseMetric):
    """3D instance segmentation evaluation metric.

    This metric is for ScanNet 3D instance segmentation tasks. For more info
    about ScanNet, please read [here](https://github.com/ScanNet/ScanNet).

    Args:
        dataset_meta (dict, optional): Provide dataset meta information.
        classes (List[str], optional): Provide dataset classes information as
            an alternative to dataset_meta.
        valid_class_ids (List[int], optional): Provide dataset valid class ids
            information as an alternative to dataset_meta.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Example:
        >>> import numpy as np
        >>> from mmeval import InstanceSegMetric
        >>> seg_valid_class_ids = (3, 4, 5)
        >>> class_labels = ('cabinet', 'bed', 'chair')
        >>> dataset_meta = dict(
        ...     seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)
        >>>
        >>> def _demo_mm_model_output(self):
        ...     n_points_list = [3300, 3000]
        ...     gt_labels_list = [[0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
        ...                       [2, 2, 2, 1, 0, 0, 0, 0, 0]]
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
        'all_ap': 0.8333333333333334,
        'all_ap_50%': 0.8333333333333334,
        'all_ap_25%': 0.8333333333333334,
        'classes': {
            'cabinet': {
                'ap': 1.0,
                'ap50%': 1.0,
                'ap25%': 1.0
            },
            'bed': {
                'ap': 0.5,
                'ap50%': 0.5,
                'ap25%': 0.5
            },
            'chair': {
                'ap': 1.0,
                'ap50%': 1.0,
                'ap25%': 1.0
            }
        }
        }
    """

    def __init__(self,
                 classes: Optional[List[str]] = None,
                 valid_class_ids: Optional[List[int]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._valid_class_ids = valid_class_ids
        self._classes = classes

    @property
    def classes(self):
        """Returns classes.

        The classes should be set during initialization, otherwise it will
        be obtained from the 'classes' field in ``self.dataset_meta``.

        Raises:
            RuntimeError: If the classes is not set.

        Returns:
            List[str]: The classes.
        """
        if self._classes is not None:
            return self._classes

        if self.dataset_meta and 'classes' in self.dataset_meta:
            self._classes = self.dataset_meta['classes']
        else:
            raise RuntimeError('The `classes` is required, and not found in '
                               f'dataset_meta: {self.dataset_meta}')
        return self._classes

    @property
    def valid_class_ids(self):
        """Returns valid class ids.

        The valid class ids should be set during initialization, otherwise
        it will be obtained from the 'seg_valid_class_ids' field in
        ``self.dataset_meta``.

        Raises:
            RuntimeError: If valid class ids is not set.

        Returns:
            List[str]: The valid class ids.
        """
        if self._classes is not None:
            return self._valid_class_ids

        if self.dataset_meta and 'seg_valid_class_ids' in self.dataset_meta:
            self._valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        else:
            raise RuntimeError(
                'The `seg_valid_class_ids` is required, and not found in '
                f'dataset_meta: {self.dataset_meta}')
        return self._valid_class_ids

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

        assert len(self.valid_class_ids) == len(self.classes)
        id_to_label = {
            self.valid_class_ids[i]: self.classes[i]
            for i in range(len(self.valid_class_ids))
        }
        preds = self.aggregate_predictions(
            masks=pred_instance_masks,
            labels=pred_instance_labels,
            scores=pred_instance_scores,
            valid_class_ids=self.valid_class_ids)
        gts = self.rename_gt(gt_semantic_masks, gt_instance_masks,
                             self.valid_class_ids)
        metrics = scannet_eval(
            preds=preds,
            gts=gts,
            options=None,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            id_to_label=id_to_label)

        return metrics

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
            assert len(
                unique
            ) < 1000, 'The nums of label in gt should not be greater than 1000'
            for i in unique:
                semantic_instance = semantic_mask[instance_mask == i]
                semantic_unique = np.unique(semantic_instance)
                assert len(semantic_unique) == 1
                if semantic_unique[0] < len(valid_class_ids):
                    instance_mask[instance_mask == i] = 1000 * valid_class_ids[
                        semantic_unique[0]] + i  # noqa: E501
            renamed_instance_masks.append(instance_mask)
        return renamed_instance_masks
