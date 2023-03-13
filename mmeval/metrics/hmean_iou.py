# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

from mmeval.core import BaseMetric
from mmeval.metrics.utils import poly_intersection, poly_iou, polys2shapely
from mmeval.utils import try_import

if TYPE_CHECKING:
    import shapely.geometry as geometry
    from shapely.geometry import Polygon
else:
    geometry = try_import('shapely.geometry')
    if geometry is not None:
        Polygon = geometry.Polygon

BATCH_POLYS = Sequence[Sequence[np.ndarray]]


def compute_hmean(accum_hit_recall: Union[int, float],
                  accum_hit_prec: Union[int, float], gt_num: int,
                  pred_num: int) -> Tuple[float, float, float]:
    """Compute hmean given hit number, ground truth number and prediction
    number.

    Args:
        accum_hit_recall (int or float): Accumulated hits for computing recall.
        accum_hit_prec (int or float): Accumulated hits for computing
            precision.
        gt_num (int): Ground truth number.
        pred_num (int): Prediction number.

    Returns:
        Tuple(float, float, float): Recall, precision and hmean.
    """

    assert isinstance(accum_hit_recall, (float, int))
    assert isinstance(accum_hit_prec, (float, int))

    assert isinstance(gt_num, int)
    assert isinstance(pred_num, int)
    assert accum_hit_recall >= 0.0
    assert accum_hit_prec >= 0.0
    assert gt_num >= 0.0
    assert pred_num >= 0.0

    if gt_num == 0:
        recall = 1.0
        precision = 0.0 if pred_num > 0 else 1.0
    else:
        recall = float(accum_hit_recall) / gt_num
        precision = 0.0 if pred_num == 0 else float(accum_hit_prec) / pred_num

    denom = recall + precision

    hmean = 0.0 if denom == 0 else (2.0 * precision * recall / denom)

    return recall, precision, hmean


class HmeanIoU(BaseMetric):
    """HmeanIoU metric.

    This method computes the hmean iou metric of polygons, and accepts
    parameters orgnaized as follows:

    - batch_pred_polygons (Sequence[Sequence[np.ndarray]]): A batch of
      prediction polygons, where each element is a sequence of
      polygons. Each polygon is represented in the form of [x1, y1,
      x2, y2, ...].
    - batch_pred_scores (Sequence): A batch of prediction scores, where each
      element is a sequence of scores.
    - batch_pred_polygons (Sequence[Sequence[np.ndarray]]): A batch of
      ground truth polygons, where each element is a sequence of
      polygons. Each polygon is represented in the form of [x1, y1,
      x2, y2, ...].
    - batch_gt_ignore_flags (Sequence): A batch of boolean flags indicating
      whether to ignore the corresponding ground truth polygon. Each element is
      a sequence of flags.

    The evaluation is done in the following steps:

    - Filter the prediction polygon:

      - Score is smaller than minimum prediction score threshold.
      - The proportion of the area that intersects with gt ignored polygon is
        greater than ignore_precision_thr.

    - Computing an M x N IoU matrix, where each element indexing
      E_mn represents the IoU between the m-th valid GT and n-th valid
      prediction.
    - Based on different prediction score threshold:

      - Obtain the ignored predictions according to prediction score.
        The filtered predictions will not be involved in the later metric
        computations.
      - Based on the IoU matrix, get the match metric according to
        ``match_iou_thr``.
      - Based on different `strategy`, accumulate the match number.

    - calculate H-mean under different prediction score threshold.

    Args:
        match_iou_thr (float): IoU threshold for a match. Defaults to 0.5.
        ignore_precision_thr (float): Precision threshold when prediction and
            gt ignored polygons are matched. Defaults to 0.5.
        pred_score_thrs (dict): Best prediction score threshold searching
            space. Defaults to dict(start=0.3, stop=0.9, step=0.1).
        strategy (str): Polygon matching strategy. Options are 'max_matching'
            and 'vanilla'. 'max_matching' refers to the optimum strategy that
            maximizes the number of matches. Vanilla strategy matches gt and
            pred polygons if both of them are never matched before. It was used
            in academia. Defaults to 'vanilla'.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import HmeanIoU
        >>> import numpy as np
        >>> hmeaniou = HmeanIoU(pred_score_thrs=dict(start=0.5, stop=0.7, step=0.1))  # noqa
        >>> gt_polygons = [[np.array([0, 0, 1, 0, 1, 1, 0, 1])]]
        >>> pred_polygons = [[
        ...     np.array([0, 0, 1, 0, 1, 1, 0, 1]),
        ...     np.array([0, 0, 1, 0, 1, 1, 0, 1]),
        ... ]]
        >>> pred_scores = [np.array([1, 0.5])]
        >>> gt_ignore_flags = [[False]]
        >>> hmeaniou(pred_polygons, pred_scores, gt_polygons, gt_ignore_flags)
        {
            0.5: {'precision': 0.5, 'recall': 1.0, 'hmean': 0.6666666666666666},  # noqa
            0.6: {'precision': 1.0, 'recall': 1.0, 'hmean': 1.0},
            'best': {'precision': 1.0, 'recall': 1.0, 'hmean': 1.0}
        }
    """

    def __init__(
        self,
        match_iou_thr: float = 0.5,
        ignore_precision_thr: float = 0.5,
        pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
        strategy: str = 'vanilla',
        **kwargs,
    ) -> None:
        if geometry is None:
            raise RuntimeError(
                'shapely is not installed, please run "pip install shapely" to'
                ' use HmeanIoUMetric.')
        super().__init__(**kwargs)
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self.pred_score_thrs = np.arange(**pred_score_thrs)
        assert strategy in ['max_matching', 'vanilla']
        self.strategy = strategy

    def add(self, batch_pred_polygons: BATCH_POLYS, batch_pred_scores: Sequence, batch_gt_polygons: BATCH_POLYS, batch_gt_ignore_flags: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Args:
            batch_pred_polygons (Sequence[Sequence[np.ndarray]]): A batch of
                prediction polygons, where each element is a sequence of
                polygons. Each polygon is represented in the form of [x1, y1,
                x2, y2, ...].
            batch_pred_scores (Sequence): A batch of
                prediction scores, where each element is a sequence of
                scores.
            batch_pred_polygons (Sequence[Sequence[np.ndarray]]): A batch of
                ground truth polygons, where each element is a sequence of
                polygons. Each polygon is represented in the form of [x1, y1,
                x2, y2, ...].
            batch_gt_ignore_flags (Sequence): A batch of
                boolean flags indicating whether to ignore the corresponding
                ground truth polygon. Each element is a sequence of flags.
        """
        for pred_polygons, pred_scores, gt_polygons, gt_ignore_flags in zip(
                batch_pred_polygons, batch_pred_scores, batch_gt_polygons,
                batch_gt_ignore_flags):

            pred_polys = polys2shapely(pred_polygons)
            pred_scores = np.array(pred_scores, dtype=np.float32)
            gt_polys = polys2shapely(gt_polygons)
            gt_ignore_flags = np.array(gt_ignore_flags, dtype=np.bool_)

            self._results.append(
                (pred_polys, pred_scores, gt_polys, gt_ignore_flags))

    def compute_metric(self, results: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Dict:  # type: ignore # yapf: disable # noqa: E501
        """Compute the metrics from processed results.

        Args:
            results (list[(np.ndarray, np.ndarray, np.ndnarray, np.ndarray]):
                The processed results of each batch.

        Returns:
            dict[float or "best", dict[str, float]]: Nested dicts as results.
            The inner dict contains the "precision", "recall", and "hmean"
            scores under different prediction score thresholds, which can be
            indexed by the corresponding threshold value from the outer dict.
            The outer dict also contains the "best" key, which is the result
            of the best hmean score.
        """

        best_eval_results = dict(hmean=-1)

        dataset_pred_num = np.zeros_like(self.pred_score_thrs)
        dataset_hit_num = np.zeros_like(self.pred_score_thrs)
        dataset_gt_num = 0

        for pred_polys, pred_scores, gt_polys, gt_ignore_flags in results:

            pred_ignore_flags = self._filter_preds(pred_polys, gt_polys,
                                                   pred_scores,
                                                   gt_ignore_flags)
            gt_num = np.sum(~gt_ignore_flags)
            pred_num = np.sum(~pred_ignore_flags)
            iou_metric = np.zeros([gt_num, pred_num])

            # Compute IoU scores amongst kept pred and gt polygons
            for pred_mat_id, pred_poly_id in enumerate(
                    self._true_indexes(~pred_ignore_flags)):
                for gt_mat_id, gt_poly_id in enumerate(
                        self._true_indexes(~gt_ignore_flags)):
                    iou_metric[gt_mat_id, pred_mat_id] = poly_iou(
                        gt_polys[gt_poly_id], pred_polys[pred_poly_id])

            pred_scores = pred_scores[~pred_ignore_flags]  # (pred_num, )
            dataset_gt_num += iou_metric.shape[0]

            # Filter out predictions by IoU threshold
            for i, pred_score_thr in enumerate(self.pred_score_thrs):
                pred_ignore_flags = pred_scores < pred_score_thr
                # get the number of matched boxes
                matched_metric = iou_metric[:, ~pred_ignore_flags] \
                    > self.match_iou_thr
                if self.strategy == 'max_matching':
                    dataset_hit_num[i] += self._max_matching(matched_metric)
                else:
                    dataset_hit_num[i] += self._vanilla_matching(
                        matched_metric)
                dataset_pred_num[i] += np.sum(~pred_ignore_flags)

        eval_results = dict()
        for i, pred_score_thr in enumerate(self.pred_score_thrs):
            recall, precision, hmean = compute_hmean(
                int(dataset_hit_num[i]), int(dataset_hit_num[i]),
                int(dataset_gt_num), int(dataset_pred_num[i]))
            eval_results[pred_score_thr] = dict(
                precision=precision, recall=recall, hmean=hmean)
            if hmean > best_eval_results['hmean']:
                best_eval_results = eval_results[pred_score_thr]  # type: ignore # yapf: disable # noqa: E501
        eval_results['best'] = best_eval_results  # type: ignore
        return eval_results

    def _max_matching(self, iou_metric: np.ndarray) -> int:
        """Max matching strategy.

        Args:
            iou_metric (np.ndarray): IoU metric matrix.

        Returns:
            int: The hits by max matching policy.
        """
        csr_matched_metric = sparse.csr_matrix(iou_metric)
        matched_preds = sparse.csgraph.maximum_bipartite_matching(
            csr_matched_metric, perm_type='row')
        # -1 denotes unmatched pred polygons
        return np.sum(matched_preds != -1).item()

    def _vanilla_matching(self, iou_metric: np.ndarray) -> int:
        """Vanilla matching strategy.

        Args:
            iou_metric (np.ndarray): IoU metric matrix.

        Returns:
            int: The hits by vanilla matching policy.
        """
        matched_gt_indexes = set()
        matched_pred_indexes = set()
        for gt_idx, pred_idx in zip(*np.nonzero(iou_metric)):
            if gt_idx in matched_gt_indexes or \
              pred_idx in matched_pred_indexes:
                continue
            matched_gt_indexes.add(gt_idx)
            matched_pred_indexes.add(pred_idx)
        return len(matched_gt_indexes)

    def _filter_preds(self, pred_polys: List['Polygon'],
                      gt_polys: List['Polygon'], pred_scores: List[float],
                      gt_ignore_flags: np.ndarray) -> np.ndarray:
        """Filter out the predictions by score threshold and whether it
        overlaps ignored gt polygons.

        Args:
            pred_polys (list[Polygon]): Pred polygons.
            gt_polys (list[Polygon]): GT polygons.
            pred_scores (list[float]): Pred scores of polygons.
            gt_ignore_flags (np.ndarray): 1D boolean array indicating
                the positions of ignored gt polygons.

        Returns:
            np.ndarray: 1D boolean array indicating the positions of ignored
            pred polygons.
        """

        # Filter out predictions based on the minimum score threshold
        pred_ignore_flags = pred_scores < self.pred_score_thrs.min()

        # Filter out pred polygons which overlaps any ignored gt polygons
        for pred_id in self._true_indexes(~pred_ignore_flags):
            for gt_id in self._true_indexes(gt_ignore_flags):
                # Match pred with ignored gt
                precision = poly_intersection(
                    gt_polys[gt_id], pred_polys[pred_id]) / (
                        pred_polys[pred_id].area + 1e-5)
                if precision > self.ignore_precision_thr:
                    pred_ignore_flags[pred_id] = True
                    break

        return pred_ignore_flags

    def _true_indexes(self, array: np.ndarray) -> np.ndarray:
        """Get indexes of True elements from a 1D boolean array."""
        return np.where(array)[0]
