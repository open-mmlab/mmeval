# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Sequence, Union

from mmeval.core.base_metric import BaseMetric
from .utils import calc_distances, distance_acc


def keypoint_pck_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                          thr: float, norm_factor: np.ndarray) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt


class PCKAccuracy(BaseMetric):
    """PCK accuracy evaluation metric, which is widely used in pose estimation.

    Calculate the pose accuracy of Percentage of Correct Keypoints (PCK) for
    each individual keypoint and the averaged accuracy across all keypoints.
    PCK metric measures the accuracy of the localization of the body joints.
    The distances between predicted positions and the ground-truth ones
    are typically normalized by the person bounding box size.
    The threshold (thr) of the normalized distance is commonly set
    as 0.05, 0.1 or 0.2 etc.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        thr(float): Threshold of PCK calculation. Defaults to 0.2.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso', which correspond
            to 'PCK', 'PCKh' and 'tPCK' respectively. Defaults to ``'bbox'``.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import PCKAccuracy
        >>> import numpy as np
        >>> num_keypoints = 15
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> predictions = [{'coords': keypoints}]
        >>> keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
        >>> bbox_size = np.random.random((1, 2)) * 10
        >>> groundtruths = [{
        ...     'coords': keypoints,
        ...     'mask': keypoints_visible,
        ...     'bbox_size': bbox_size,
        ... }]
        >>> pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
        >>> pck_metric(predictions, groundtruths)
        OrderedDict([('PCK@0.5', 1.0)])
    """

    def __init__(self,
                 thr: float = 0.2,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.thr = thr
        if isinstance(norm_item, str):
            norm_item = [norm_item]
        self.norm_item = norm_item

        allow_normalized_items = ['bbox', 'head', 'torso']
        for item in self.norm_item:
            if item not in allow_normalized_items:
                raise KeyError(
                    f'The normalized item {item} is not supported by '
                    f"{self.__class__.__name__}. Should be one of 'bbox', "
                    f"'head', 'torso', but got {item}.")

    def add(self, predictions: List[Dict], groundtruths: List[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of predictions and groundtruths and add the
        intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): Predictions from the model.
                Each prediction dict has the following keys:

                - coords (np.ndarray, [1, K, D]): predicted keypoints
                  coordinates

            groundtruths (Sequence[dict]): The ground truth labels.
                Each groundtruth dict has the following keys:

                - coords (np.ndarray, [1, K, D]): ground truth keypoints
                  coordinates
                - mask (np.ndarray, [1, K]): ground truth keypoints_visible
                - bbox_size (np.ndarray, optional, [1, 2]): ground truth
                  bbox size
                - head_size (np.ndarray, optional, [1, 2]): ground truth
                  head size
                - torso_size (np.ndarray, optional, [1, 2]): ground truth
                  torso size
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
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

        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate([gt['bbox_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"bbox_size"``)...')

            _, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                              self.thr, norm_size_bbox)
            metric_results[f'PCK@{self.thr}'] = pck

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([gt['head_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"head_size"``)...')

            _, pckh, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                               self.thr, norm_size_head)
            metric_results[f'PCKh@{self.thr}'] = pckh

        if 'torso' in self.norm_item:
            norm_size_torso = np.concatenate([gt['torso_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"torso_size"``)...')

            _, tpck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                               self.thr, norm_size_torso)
            metric_results[f'tPCK@{self.thr}'] = tpck

        return metric_results


class MpiiPCKAccuracy(PCKAccuracy):
    """PCKh accuracy evaluation metric for MPII dataset.

    Calculate the pose accuracy of Percentage of Correct Keypoints (PCK) for
    each individual keypoint and the averaged accuracy across all keypoints.
    PCK metric measures accuracy of the localization of the body joints.
    The distances between predicted positions and the ground-truth ones
    are typically normalized by the person bounding box size.
    The threshold (thr) of the normalized distance is commonly set
    as 0.05, 0.1 or 0.2 etc.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        thr(float): Threshold of PCK calculation. Defaults to 0.5.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso', which correspond
            to 'PCK', 'PCKh' and 'tPCK' respectively. Defaults to ``'head'``.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import MpiiPCKAccuracy
        >>> import numpy as np
        >>> num_keypoints = 16
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> predictions = [{'coords': keypoints}]
        >>> keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
        >>> head_size = np.random.random((1, 2)) * 10
        >>> groundtruths = [{
        ...     'coords': keypoints + 1.0,
        ...     'mask': keypoints_visible,
        ...     'head_size': head_size,
        ... }]
        >>> mpii_pckh_metric = MpiiPCKAccuracy(thr=0.3, norm_item='head')
        >>> mpii_pckh_metric(predictions, groundtruths)
        OrderedDict([('Head', 100.0), ('Shoulder', 100.0), ('Elbow', 100.0),
        ('Wrist', 100.0), ('Hip', 100.0), ('Knee', 100.0), ('Ankle', 100.0),
        ('PCKh', 100.0), ('PCKh@0.1', 100.0)])
    """

    def __init__(self,
                 thr: float = 0.5,
                 norm_item: Union[str, Sequence[str]] = 'head',
                 **kwargs) -> None:
        super().__init__(thr=thr, norm_item=norm_item, **kwargs)

    def compute_metric(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # split gt and prediction list
        preds, gts = zip(*results)

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([pred['coords'] for pred in preds])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([gt['coords'] for gt in gts])
        # mask: [N, K]
        mask = np.concatenate([gt['mask'] for gt in gts])

        # MPII uses matlab format, gt index is 1-based,
        # convert 0-based index to 1-based index
        pred_coords = pred_coords + 1.0

        metric_results = super().compute_metric(results)

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([gt['head_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"head_size"``)...')

            pck_p, _, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                                self.thr, norm_size_head)

            jnt_count = np.sum(mask, axis=0)
            PCKh = 100. * pck_p

            rng = np.arange(0, 0.5 + 0.01, 0.01)
            pckAll = np.zeros((len(rng), 16), dtype=np.float32)

            for r, threshold in enumerate(rng):
                _pck, _, _ = keypoint_pck_accuracy(pred_coords, gt_coords,
                                                   mask, threshold,
                                                   norm_size_head)
                pckAll[r, :] = 100. * _pck

            PCKh = np.ma.array(PCKh, mask=False)
            PCKh.mask[6:8] = True

            jnt_count = np.ma.array(jnt_count, mask=False)
            jnt_count.mask[6:8] = True
            jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

            # dataset_joints_idx:
            #   head 9
            #   lsho 13  rsho 12
            #   lelb 14  relb 11
            #   lwri 15  rwri 10
            #   lhip 3   rhip 2
            #   lkne 4   rkne 1
            #   lank 5   rank 0
            stats = {
                'Head': PCKh[9],
                'Shoulder': 0.5 * (PCKh[13] + PCKh[12]),
                'Elbow': 0.5 * (PCKh[14] + PCKh[11]),
                'Wrist': 0.5 * (PCKh[15] + PCKh[10]),
                'Hip': 0.5 * (PCKh[3] + PCKh[2]),
                'Knee': 0.5 * (PCKh[4] + PCKh[1]),
                'Ankle': 0.5 * (PCKh[5] + PCKh[0]),
                'PCKh': np.sum(PCKh * jnt_ratio),
                'PCKh@0.1': np.sum(pckAll[10, :] * jnt_ratio)
            }

            del metric_results[f'PCKh@{self.thr}']
            for stats_name, stat in stats.items():
                metric_results[stats_name] = stat

        return metric_results


class JhmdbPCKAccuracy(PCKAccuracy):
    """PCK accuracy evaluation metric for Jhmdb dataset.

    Calculate the pose accuracy of Percentage of Correct Keypoints (PCK) for
    each individual keypoint and the averaged accuracy across all keypoints.
    PCK metric measures accuracy of the localization of the body joints.
    The distances between predicted positions and the ground-truth ones
    are typically normalized by the person bounding box size.
    The threshold (thr) of the normalized distance is commonly set
    as 0.05, 0.1 or 0.2 etc.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        thr(float): Threshold of PCK calculation. Defaults to 0.5.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso', which correspond
            to 'PCK', 'PCKh' and 'tPCK' respectively. Defaults to ``'bbox'``.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import JhmdbPCKAccuracy
        >>> import numpy as np
        >>> num_keypoints = 15
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> predictions = [{'coords': keypoints}]
        >>> keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
        >>> torso_size = np.random.random((1, 2)) * 10
        >>> groundtruths = [{
        ...     'coords': keypoints,
        ...     'mask': keypoints_visible,
        ...     'torso_size': torso_size,
        ... }]
        >>> jhmdb_pckh_metric = JhmdbPCKAccuracy(thr=0.2, norm_item='torso')
        >>> jhmdb_pckh_metric(predictions, groundtruths)
        OrderedDict([('Head tPCK', 1.0), ('Sho tPCK', 1.0), ('Elb tPCK', 1.0),
        ('Wri tPCK', 1.0), ('Hip tPCK', 1.0), ('Knee tPCK', 1.0),
        ('Ank tPCK', 1.0), ('Mean tPCK', 1.0)])
    """

    def __init__(self,
                 thr: float = 0.5,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 **kwargs) -> None:
        super().__init__(thr=thr, norm_item=norm_item, **kwargs)

    def compute_metric(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # split gt and prediction list
        preds, gts = zip(*results)

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([pred['coords'] for pred in preds])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([gt['coords'] for gt in gts])
        # mask: [N, K]
        mask = np.concatenate([gt['mask'] for gt in gts])

        metric_results = super().compute_metric(results)

        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate([gt['bbox_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"bbox_size"``)...')

            pck_p, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                                  self.thr, norm_size_bbox)
            metric_results[f'PCK@{self.thr}'] = pck

            stats = {
                'Head': pck_p[2],
                'Sho': 0.5 * pck_p[3] + 0.5 * pck_p[4],
                'Elb': 0.5 * pck_p[7] + 0.5 * pck_p[8],
                'Wri': 0.5 * pck_p[11] + 0.5 * pck_p[12],
                'Hip': 0.5 * pck_p[5] + 0.5 * pck_p[6],
                'Knee': 0.5 * pck_p[9] + 0.5 * pck_p[10],
                'Ank': 0.5 * pck_p[13] + 0.5 * pck_p[14],
                'Mean': pck
            }

            for stats_name, stat in stats.items():
                metric_results[f'{stats_name} PCK'] = stat

        if 'torso' in self.norm_item:
            norm_size_torso = np.concatenate([gt['torso_size'] for gt in gts])

            self.logger.info(f'Evaluating {self.__class__.__name__} '
                             f'(normalized by ``"torso_size"``)...')

            pck_p, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                                  self.thr, norm_size_torso)

            stats = {
                'Head': pck_p[2],
                'Sho': 0.5 * pck_p[3] + 0.5 * pck_p[4],
                'Elb': 0.5 * pck_p[7] + 0.5 * pck_p[8],
                'Wri': 0.5 * pck_p[11] + 0.5 * pck_p[12],
                'Hip': 0.5 * pck_p[5] + 0.5 * pck_p[6],
                'Knee': 0.5 * pck_p[9] + 0.5 * pck_p[10],
                'Ank': 0.5 * pck_p[13] + 0.5 * pck_p[14],
                'Mean': pck
            }

            del metric_results[f'tPCK@{self.thr}']
            for stats_name, stat in stats.items():
                metric_results[f'{stats_name} tPCK'] = stat

        return metric_results
