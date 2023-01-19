# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
from collections import OrderedDict
from typing import List, Sequence, Union, Dict

from mmeval.core.base_metric import BaseMetric

logger = logging.getLogger(__name__)


def pairwise_temporal_iou(candidate_segments,
                          target_segments,
                          calculate_overlap_self=False):
    """Compute intersection over union between segments.

    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            ``[init, end]/[m x 2:=[init, end]]``.
        target_segments (np.ndarray): 2-dim array in format
            ``[n x 2:=[init, end]]``.
        calculate_overlap_self (bool): Whether to calculate overlap_self
            (union / candidate_length) or not. Default: False.
    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
        t_overlap_self (np.ndarray, optional): 1-dim array [n] /
            2-dim array [n x m] with overlap_self, returns when
            calculate_overlap_self is True.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    num_targets = target_segments.shape[0]
    num_candidates = candidate_segments.shape[0]
    t_iou = np.empty((num_targets, num_candidates), dtype=np.float32)
    if calculate_overlap_self:
        t_overlap_self = np.empty((num_targets, num_candidates),
                                  dtype=np.float32)

    for i in range(num_candidates):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)
        if calculate_overlap_self:
            candidate_length = candidate_segment[1] - candidate_segment[0]
            t_overlap_self[:, i] = (
                segments_intersection.astype(float) / candidate_length)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)
    if calculate_overlap_self:
        if candidate_segments_ndim == 1:
            t_overlap_self = np.squeeze(t_overlap_self, axis=1)
        return t_iou, t_overlap_self

    return t_iou


def average_recall_at_avg_proposals(ground_truth,
                                    proposals,
                                    total_num_proposals,
                                    max_avg_proposals=None,
                                    temporal_iou_thresholds=np.linspace(
                                        0.5, 0.95, 10)):
    """Computes the average recall given an average number (percentile) of
    proposals per video.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
        proposals (dict): Dict containing the proposal instances.
        total_num_proposals (int): Total number of proposals in the
            proposal dict.
        max_avg_proposals (int | None): Max number of proposals for one video.
            Default: None.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.
    Returns:
        tuple([np.ndarray, np.ndarray, np.ndarray, float]):
            (recall, average_recall, proposals_per_video, auc)
            In recall, ``recall[i,j]`` is recall at i-th temporal_iou threshold
            at the j-th average number (percentile) of average number of
            proposals per video. The average_recall is recall averaged
            over a list of temporal_iou threshold (1D array). This is
            equivalent to ``recall.mean(axis=0)``. The ``proposals_per_video``
            is the average number of proposals per video. The auc is the area
            under ``AR@AN`` curve.
    """

    total_num_videos = len(ground_truth)

    if not max_avg_proposals:
        max_avg_proposals = float(total_num_proposals) / total_num_videos

    ratio = (max_avg_proposals * float(total_num_videos) / total_num_proposals)

    # For each video, compute temporal_iou scores among the retrieved proposals
    score_list = []
    total_num_retrieved_proposals = 0
    for video_id in ground_truth:
        # Get proposals for this video.
        proposals_video_id = proposals[video_id]
        this_video_proposals = proposals_video_id[:, :2]
        # Sort proposals by score.
        sort_idx = proposals_video_id[:, 2].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :].astype(
            np.float32)

        # Get ground-truth instances associated to this video.
        ground_truth_video_id = ground_truth[video_id]
        this_video_ground_truth = ground_truth_video_id[:, :2].astype(
            np.float32)
        if this_video_proposals.shape[0] == 0:
            score_list.append(np.zeros((this_video_ground_truth.shape[0], 1)))
            continue

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(
                this_video_ground_truth, axis=0)

        num_retrieved_proposals = np.minimum(
            int(this_video_proposals.shape[0] * ratio),
            this_video_proposals.shape[0])
        total_num_retrieved_proposals += num_retrieved_proposals
        this_video_proposals = this_video_proposals[:
                                                    num_retrieved_proposals, :]

        # Compute temporal_iou scores.
        t_iou = pairwise_temporal_iou(this_video_proposals,
                                      this_video_ground_truth)
        score_list.append(t_iou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_list = np.arange(1, 101) / 100.0 * (
        max_avg_proposals * float(total_num_videos) /
        total_num_retrieved_proposals)
    matches = np.empty((total_num_videos, pcn_list.shape[0]))
    positives = np.empty(total_num_videos)
    recall = np.empty((temporal_iou_thresholds.shape[0], pcn_list.shape[0]))
    # Iterates over each temporal_iou threshold.
    for ridx, temporal_iou in enumerate(temporal_iou_thresholds):
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_list):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum temporal_iou threshold.
            true_positives_temporal_iou = score >= temporal_iou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum(
                (score.shape[1] * pcn_list).astype(np.int32), score.shape[1])

            for j, num_retrieved_proposals in enumerate(pcn_proposals):
                # Compute the number of matches
                # for each percentage of the proposals
                matches[i, j] = np.count_nonzero(
                    (true_positives_temporal_iou[:, :num_retrieved_proposals]
                     ).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_list * (
        float(total_num_retrieved_proposals) / total_num_videos)
    # Get AUC
    area_under_curve = np.trapz(avg_recall, proposals_per_video)
    auc = 100. * float(area_under_curve) / proposals_per_video[-1]
    return recall, avg_recall, proposals_per_video, auc


class ActivityNetAR(BaseMetric):
    """ActivityNet evaluation metric.
    ActivityNet-1.3 dataset: http://activity-net.org/download.html.

    This metric computes AR under different Average Number of proposals (AR =
    1, 5, 10, 100) as AR@1, AR@5, AR@10 and AR@100, and calculate the Area 
    under the AR vs. AN curve (AUC) as metrics.

    temporal_iou_thresholds (np.array or List): the list of temporal iOU
        thresholds. Defaults to np.linspace(0.5, 0.95, 10).
    max_avg_proposals (int): the maximun of Average Number of proposals.
        Defaults to 100.
     **kwargs: Keyword parameters passed to :class:`BaseMetric`.


    Examples:
        >>> from mmeval import ActivityNetAR
        >>> anet_metric = ActivityNetAR()
        >>> predictions = [
        >>>                {
        >>>                 'video_name': 'v_--1DO2V4K74',
        >>>                 'annotations': [
        >>>                                 {
        >>>                                  'segment': [30.02, 180], 
        >>>                                  'label': 1
        >>>                                  }
        >>>                                 ],
        >>>                 'proposal_list': [
        >>>                     {'segment': [53, 68], 'score': 0.21},
        >>>                     {'segment': [38, 110], 'score': 0.54},
        >>>                     {'segment': [65, 128], 'score': 0.95},
        >>>                     {'segment': [69, 93], 'score': 0.98},
        >>>                     {'segment': [99, 147], 'score': 0.84},
        >>>                     {'segment': [28, 96], 'score': 0.84},
        >>>                     {'segment': [18, 92], 'score': 0.22},
        >>>                     {'segment': [40, 66], 'score': 0.36},
        >>>                     {'segment': [14, 29], 'score': 0.75},
        >>>                     {'segment': [67, 105], 'score': 0.25},
        >>>                     {'segment': [2, 7], 'score': 0.94},
        >>>                     {'segment': [25, 112], 'score': 0.49},
        >>>                     {'segment': [7, 83], 'score': 0.9},
        >>>                     {'segment': [75, 159], 'score': 0.42},
        >>>                     {'segment': [99, 176], 'score': 0.62},
        >>>                     {'segment': [89, 186], 'score': 0.56},
        >>>                     {'segment': [50, 200], 'score': 0.5}
        >>>                     ]
        >>>                 }
        >>>                ]
        >>> anet_metric(predictions)
        OrderedDict([('auc', 54.2), ('AR@1', 0.0), ('AR@5', 0.0), ('AR@10', 0.2), ('AR@100', 0.6)])
    """

    def __init__(self,
                 temporal_iou_thresholds: Union[np.array, List] = np.linspace(
                     0.5, 0.95, 10),
                 max_avg_proposals: int = 100,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.temporal_iou_thresholds = np.array(temporal_iou_thresholds)
        self.max_avg_proposals = max_avg_proposals
        self.ground_truth: Dict[str, np.array] = {}

    def add(self, predictions: Sequence[dict]) -> None:  # type: ignore
        """Add detection results to the results list.

        Args:
            predictions (Sequence[dict]): A list of prediction dict which
                contains the following keys:

                - `video_name`: The name of the video, e.g., `v_--1DO2V4K74`.
                   The name should be of the format of `v_{video_id}`.
                - `annotations`: A list of annotations. Each annotation is a
                  dict with keys `segment` and `label`. The value of `segment`
                  is `[start, end]`. The value of `label` is an int label.
                - `proposal_list`: A list of proposals. Each proposal is a
                  dict with keys `segment` and `score`. The value of `segment`
                  is `[start, end]`. The value of `score` is a float number.
        """
        for pred in predictions:
            self._results.append(pred)
            video_id = pred['video_name'][2:]
            ground_truth = []
            for ann in pred['annotations']:
                t_start, t_end = ann['segment']
                label = ann['label']
                ground_truth.append([t_start, t_end, label])
            self.ground_truth[video_id] = np.array(ground_truth)

    def compute_metric(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        eval_results = OrderedDict()
        proposal, num_proposals = self._import_proposals(results)

        recall, _, _, auc = average_recall_at_avg_proposals(
            self.ground_truth,
            proposal,
            num_proposals,
            max_avg_proposals=self.max_avg_proposals,
            temporal_iou_thresholds=self.temporal_iou_thresholds)
        eval_results['auc'] = auc
        eval_results['AR@1'] = np.mean(recall[:, 0])
        eval_results['AR@5'] = np.mean(recall[:, 4])
        eval_results['AR@10'] = np.mean(recall[:, 9])
        eval_results['AR@100'] = np.mean(recall[:, 99])

        return eval_results

    @staticmethod
    def _import_proposals(results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name'][2:]
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals
