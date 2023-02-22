# Copyright (c) OpenMMLab. All rights reserved.
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
        >>> def _demo_mm_model_output():
        >>>     n_points_list = [3300, 3000]
        >>>     gt_labels_list = [[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 0],
        >>>                       [1, 1, 2, 1, 2, 2, 0, 0, 0, 0, 1]]
        >>>     predictions = []
        >>>     groundtruths = []
        >>>
        >>>     for idx, points_num in enumerate(n_points_list):
        >>>         points = np.ones(points_num) * -1
        >>>         gt = np.ones(points_num)
        >>>         info = {}
        >>>         for ii, i in enumerate(gt_labels_list[idx]):
        >>>             i = seg_valid_class_ids[i]
        >>>             points[ii * 300:(ii + 1) * 300] = ii
        >>>             gt[ii * 300:(ii + 1) * 300] = i * 1000 + ii
        >>>             info[f"{idx}_{ii}"] = {
        >>>                 'mask': (points == ii),
        >>>                 'label_id': i,
        >>>                 'conf': 0.99
        >>>             }
        >>>         predictions.append(info)
        >>>         groundtruths.append(gt)
        >>>
        >>>     return predictions, groundtruths
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
        if self._valid_class_ids is not None:
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
                representing a detection result. The dict has multiple keys,
                each key represents the name of the instance, and the value
                is also a dict, with following keys:

                - mask (array): Predicted instance masks.
                - label_id (int): Predicted instance labels.
                - conf (float): Predicted instance scores.

            groundtruths (Sequence[array]): A sequence of array. Each array
                represents a groundtruths for an image.
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
        preds = []
        gts = []
        for pred, gt in results:
            preds.append(pred)
            gts.append(gt)

        assert len(self.valid_class_ids) == len(self.classes)
        id_to_label = {
            self.valid_class_ids[i]: self.classes[i]
            for i in range(len(self.valid_class_ids))
        }

        metrics = scannet_eval(
            preds=preds,
            gts=gts,
            options=None,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            id_to_label=id_to_label)

        return metrics
