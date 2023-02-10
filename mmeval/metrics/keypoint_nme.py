# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from .utils import calc_distances

logger = logging.getLogger(__name__)


def keypoint_nme_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                          normalize_factor: np.ndarray) -> float:
    """Calculate the normalized mean error (NME).
    Note:
        - instance number: N
        - keypoint number: K
    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize_factor (np.ndarray[N, 2]): Normalization factor.
    Returns:
        float: normalized mean error
    """
    distances = calc_distances(pred, gt, mask, normalize_factor)
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


class KeypointNME(BaseMetric):
    """NME evaluation metric.

    Calculate the normalized mean error (NME) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        norm_mode (str): The normalization mode. There are two valid modes:
            `'use_norm_item'` and `'keypoint_distance'`.
            When set as `'use_norm_item'`, should specify the argument
            `norm_item`, which represents the item in the datainfo that
            will be used as the normalization factor.
            When set as `'keypoint_distance'`, should specify the argument
            `keypoint_indices` that are used to calculate the keypoint
            distance as the normalization factor.
        norm_item (str, optional): The item used as the normalization factor.
            For example, `'box_size'` in `'AFLWDataset'`. Only valid when
            ``norm_mode`` is ``use_norm_item``.
            Default: ``None``.
        keypoint_indices (Sequence[int], optional): The keypoint indices used
            to calculate the keypoint distance as the normalization factor.
            Only valid when ``norm_mode`` is ``keypoint_distance``.
            If set as None, will use the default ``keypoint_indices`` in
            `DEFAULT_KEYPOINT_INDICES` for specific datasets, else use the
            given ``keypoint_indices`` of the dataset. Default: ``None``.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """
    DEFAULT_KEYPOINT_INDICES = {
        # horse10: corresponding to `nose` and `eye` keypoints
        'horse10': [0, 1],
        # 300w: corresponding to `right-most` and `left-most` eye keypoints
        '300w': [36, 45],
        # coco_wholebody_face corresponding to `right-most` and `left-most`
        # eye keypoints
        'coco_wholebody_face': [36, 45],
        # cofw: corresponding to `right-most` and `left-most` eye keypoints
        'cofw': [8, 9],
        # wflw: corresponding to `right-most` and `left-most` eye keypoints
        'wflw': [60, 72],
    }

    def __init__(self,
                 norm_mode: str,
                 norm_item: Optional[str] = None,
                 keypoint_indices: Optional[Sequence[int]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        allowed_norm_modes = ['use_norm_item', 'keypoint_distance']
        if norm_mode not in allowed_norm_modes:
            raise KeyError("`norm_mode` should be 'use_norm_item' or "
                           f"'keypoint_distance', but got {norm_mode}.")
        self.norm_mode = norm_mode
        if self.norm_mode == 'use_norm_item' and not norm_item:
            raise KeyError('`norm_mode` is set to `"use_norm_item"`, '
                           'please specify the `norm_item` in the '
                           'datainfo used as the normalization factor.')
        self.norm_item = norm_item
        self.keypoint_indices = keypoint_indices

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict.
            groundtruths (Sequence[dict]): The ground truth labels.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            if self.norm_item:
                if self.norm_item == 'bbox_size':
                    assert 'bboxes' in groundtruth, 'The ground truth data ' \
                        'info do not have the item ``bboxes`` for expected ' \
                        'normalized_item ``"bbox_size"``.'
                else:
                    assert self.norm_item in groundtruth, f'The ground ' \
                        f'truth data info do not have the expected ' \
                        f'normalized factor "{self.norm_item}".'
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'   # noqa: E501
            self._results.append((prediction, groundtruth))

    def compute_metric(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are the corresponding results.
        """
        # split gt and prediction list
        preds, gts = zip(*results)

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([pred['coords'] for pred in preds])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([gt['coords'] for gt in gts])
        # mask: [N, K]
        mask = np.concatenate([gt['mask'] for gt in gts])

        metric_results: OrderedDict = OrderedDict()

        if self.norm_mode == 'use_norm_item':
            normalize_factor_ = np.concatenate(
                [gt[self.norm_item] for gt in gts])
            # normalize_factor: [N, 2]
            normalize_factor = np.tile(normalize_factor_, [1, 2])
            nme = keypoint_nme_accuracy(pred_coords, gt_coords, mask,
                                        normalize_factor)
            metric_results['NME'] = nme
        else:
            if self.keypoint_indices is None:
                # use default keypoint_indices in some datasets
                dataset_name = self.dataset_meta['dataset_name']
                if dataset_name not in self.DEFAULT_KEYPOINT_INDICES:
                    raise KeyError(
                        '`norm_mode` is set to `keypoint_distance`, and the '
                        'keypoint_indices is set to None, can not find the '
                        'keypoint_indices in `DEFAULT_KEYPOINT_INDICES`, '
                        'please specify `keypoint_indices` appropriately.')
                self.keypoint_indices = self.DEFAULT_KEYPOINT_INDICES[
                    dataset_name]
            else:
                assert len(self.keypoint_indices) == 2, 'The keypoint '\
                    'indices used for normalization should be a pair.'
                keypoint_id2name = self.dataset_meta['keypoint_id2name']
                dataset_name = self.dataset_meta['dataset_name']
                for idx in self.keypoint_indices:
                    assert idx in keypoint_id2name, f'The {dataset_name} '\
                        f'dataset does not contain the required '\
                        f'{idx}-th keypoint.'
            # normalize_factor: [N, 2]
            normalize_factor = self._get_normalize_factor(gt_coords=gt_coords)
            nme = keypoint_nme_accuracy(pred_coords, gt_coords, mask,
                                        normalize_factor)
            metric_results['NME'] = nme

        return metric_results

    def _get_normalize_factor(self, gt_coords: np.ndarray) -> np.ndarray:
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used.
        Args:
            gt_coords (np.ndarray[N, K, 2]): Groundtruth keypoint coordinates.
        Returns:
            np.ndarray[N, 2]: normalized factor
        """
        idx1, idx2 = self.keypoint_indices

        interocular = np.linalg.norm(
            gt_coords[:, idx1, :] - gt_coords[:, idx2, :],
            axis=1,
            keepdims=True)

        return np.tile(interocular, [1, 2])
