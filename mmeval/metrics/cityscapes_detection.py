# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

from mmeval.core.base_metric import BaseMetric

try:
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa: E501
    HAS_CITYSCAPESAPI = True
except ImportError:
    HAS_CITYSCAPESAPI = False

default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)


class CityScapesDetection(BaseMetric):
    """CityScapes metric for instance segmentation.

    Args:
        outfile_prefix (str): The prefix of txt and png files. It is the
            saving path of txt and png file, e.g. "a/b/prefix".
        seg_prefix (str, optional): Path to the directory which contains the
            cityscapes instance segmentation masks. It's necessary when
            training and validation. It could be None when infer on test
            dataset. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        keep_results (bool): Whether to keep the results. When ``format_only``
            is True, ``keep_results`` must be True. Defaults to False.
        keep_gt_json (bool): Whether to keep the json file during computing
            the CityScapes metric. It will load the json file on the next
            CSEval.evaluateImgLists without recalculating. Defaults to False.
        classwise (bool): Whether to return the computed  results of each
            class. Defaults to False.
        logger (Logger, optional): logger used to record messages. When set to
            ``None``, the default logger will be used.
            Defaults to None.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 outfile_prefix: str,
                 seg_prefix: Optional[str] = None,
                 format_only: bool = False,
                 keep_results: bool = False,
                 keep_gt_json: bool = False,
                 classwise: bool = False,
                 logger=None,
                 **kwargs):

        if not HAS_CITYSCAPESAPI:
            raise RuntimeError('Failed to import `cityscapesscripts`.'
                               'Please try to install official '
                               'cityscapesscripts by '
                               '"pip install cityscapesscripts"')
        super().__init__(**kwargs)

        assert outfile_prefix is not None, 'outfile_prefix must be not None'

        if format_only:
            assert keep_results and keep_gt_json, '`keep_results` and '
            '`keep_gt_json` must be True when `format_only` is True'
        else:
            assert seg_prefix is not None, '`seg_prefix` is necessary when '
            'computing the CityScapes metrics'

        self.format_only = format_only
        self.keep_results = keep_results
        self.seg_prefix = seg_prefix
        self.keep_gt_json = keep_gt_json

        self.classwise = classwise
        self.outfile_prefix = osp.abspath(outfile_prefix)
        os.makedirs(outfile_prefix, exist_ok=True)

        self.logger = default_logger if logger is None else logger

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - img_id (int): Image id.
                - bboxes (numpy.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, ), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
                - masks (list[RLE], optional): The predicted masks.
                - mask_scores (np.array, optional): Shape (N, ), the predicted
                  scores of masks.

            groundtruths (Sequence[dict]): A sequence of dict. If load from
                `ann_file`, the dict inside can be empty. Else, each dict
                represents a groundtruths for an image, with the following
                keys:

                - img_id (int): Image id.
                - width (int): The width of the image.
                - height (int): The height of the image.
                - bboxes (numpy.ndarray): Shape (K, 4), the ground truth
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - labels (numpy.ndarray): Shape (K, ), the ground truth
                  labels of bounding boxes.
                - masks (list[RLE], optional): The predicted masks.
                - ignore_flags (numpy.ndarray, optional): Shape (K, ),
                  the ignore flags.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'   # noqa: E501
            self._results.append((prediction, groundtruth))

    def compute_metric(self, results: list) -> Dict[str, Union[float, list]]:
        """Compute the CityScapes metrics.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        eval_results: OrderedDict = OrderedDict()

        if self.format_only:
            self.logger.info(
                f'Results are saved in {osp.dirname(self.outfile_prefix)}')
            return eval_results
        gt_instances_file = osp.join(self.outfile_prefix, 'gtInstances.json')
        # split gt and prediction list
        preds, gts = zip(*results)
        CSEval.args.cityscapesPath = osp.join(self.seg_prefix, '..', '..')  # type: ignore # yapf: disable # noqa: E501
        CSEval.args.predictionPath = self.outfile_prefix
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = gt_instances_file

        groundTruthImgList = [gt['file_name'] for gt in gts]
        predictionImgList = [pred['pred_txt'] for pred in preds]
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        # If gt_instances_file exists, CHEval will load it on the next
        # CSEval.evaluateImgLists without recalculating.
        if not self.keep_gt_json:
            os.remove(gt_instances_file)

        if not self.keep_results:
            shutil.rmtree(self.outfile_prefix)

        map = float(CSEval_results['allAp'])
        map_50 = float(CSEval_results['allAp50%'])

        eval_results['mAP'] = map
        eval_results['AP50'] = map_50

        results_list = ('mAP', f'{round(map * 100, 2)}',
                        f'{round(map_50 * 100, 2)}')

        if self.classwise:
            results_per_category = []
            for category, aps in CSEval_results['classes'].items():
                eval_results[f'{category}_ap'] = float(aps['ap'])
                eval_results[f'{category}_ap50'] = float(aps['ap50%'])
                results_per_category.append(
                    (f'{category}', f'{round(float(aps["ap"]) * 100, 2)}',
                     f'{round(float(aps["ap50%"]) * 100, 2)}'))
            results_per_category.append(results_list)
            eval_results['results_list'] = results_per_category
        else:
            eval_results['results_list'] = [results_list]

        return eval_results
