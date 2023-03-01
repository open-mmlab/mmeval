# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
import shutil
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

from mmeval.core.base_metric import BaseMetric

try:
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa: E501
    import cityscapesscripts.helpers.labels as CSLabels
    HAS_CITYSCAPESAPI = True
except ImportError:
    HAS_CITYSCAPESAPI = False

try:
    from mmcv import imwrite
    HAS_MMCV = True
except ImportError:
    HAS_MMCV = False

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

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import os.path as osp
        >>> import tempfile
        >>> from PIL import Image
        >>> from mmeval import CityScapesDetection
        >>>
        >>> tmp_dir = tempfile.TemporaryDirectory()
        >>> dataset_metas = {
        ...     'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
        ...                 'motorcycle', 'bicycle')
        ... }
        >>> seg_prefix = osp.join(tmp_dir.name, 'cityscapes', 'gtFine', 'val')
        >>> os.makedirs(seg_prefix, exist_ok=True)
        >>> cityscapes_det_metric = CityScapesDetection(
        ...     dataset_meta=dataset_metas,
        ...     outfile_prefix=osp.join(tmp_dir.name, 'test'),
        ...     seg_prefix=seg_prefix,
        ...     keep_results=False,
        ...     keep_gt_json=False,
        ...     classwise=True)
        >>>
        >>> def _gen_fake_datasamples(seg_prefix):
        ...     city = 'lindau'
        ...     os.makedirs(osp.join(seg_prefix, city), exist_ok=True)
        ...
        ...     sequenceNb = '000000'
        ...     frameNb1 = '000019'
        ...     img_name1 = f'{city}_{sequenceNb}_{frameNb1}_gtFine_instanceIds.png'
        ...     img_path1 = osp.join(seg_prefix, city, img_name1)
        ...     basename1 = osp.splitext(osp.basename(img_path1))[0]
        ...     masks1 = np.zeros((20, 20), dtype=np.int32)
        ...     masks1[:10, :10] = 24 * 1000
        ...     Image.fromarray(masks1).save(img_path1)
        ...
        ...     dummy_mask1 = np.zeros((1, 20, 20), dtype=np.uint8)
        ...     dummy_mask1[:, :10, :10] = 1
        ...     prediction1 = {
        ...         'basename': basename1,
        ...         'mask_scores': np.array([1.0]),
        ...         'labels': np.array([0]),
        ...         'masks': dummy_mask1
        ...     }
        ...     groundtruth1 = {
        ...         'file_name': img_path1
        ...     }
        ...
        ...     frameNb2 = '000020'
        ...     img_name2 = f'{city}_{sequenceNb}_{frameNb2}_gtFine_instanceIds.png'
        ...     img_path2 = osp.join(seg_prefix, city, img_name2)
        ...     basename2 = osp.splitext(osp.basename(img_path2))[0]
        ...     masks2 = np.zeros((20, 20), dtype=np.int32)
        ...     masks2[:10, :10] = 24 * 1000 + 1
        ...     Image.fromarray(masks2).save(img_path2)
        ...
        ...     dummy_mask2 = np.zeros((1, 20, 20), dtype=np.uint8)
        ...     dummy_mask2[:, :10, :10] = 1
        ...     prediction2 = {
        ...         'basename': basename2,
        ...         'mask_scores': np.array([0.98]),
        ...         'labels': np.array([1]),
        ...         'masks': dummy_mask1
        ...     }
        ...     groundtruth2 = {
        ...         'file_name': img_path2
        ...     }
        ...     return [prediction1, prediction2], [groundtruth1, groundtruth2]
        >>>
        >>> predictions, groundtruths = _gen_fake_datasamples(seg_prefix)
        >>> cityscapes_det_metric(predictions=predictions, groundtruths=groundtruths)  # doctest: +ELLIPSIS  # noqa: E501
        {'mAP': ..., 'AP50': ...}
        >>> tmp_dir.cleanup()
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
        if not HAS_MMCV:
            raise RuntimeError('Failed to import `mmcv`.'
                               'Please try to install official '
                               'mmcv by `pip install "mmcv>=2.0.0rc4"`')
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

        self.classes = None
        self.logger = default_logger if logger is None else logger

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - basename (str): The image name.
                - masks (numpy.ndarray): Shape (N, H, W), the predicted masks.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
                - mask_scores (np.array, optional): Shape (N, ), the predicted
                  scores of masks.

            groundtruths (Sequence[dict]): A sequence of dict. If load from
                `ann_file`, the dict inside can be empty. Else, each dict
                represents a groundtruths for an image, with the following
                keys:

                - file_name (str): The absolute path of groundtruth image.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'   # noqa: E501
            prediction = self._process_prediction(prediction)
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

        # remove `matches.json` file
        file_path = osp.join(os.getcwd(), 'matches.json')
        if osp.exists(file_path):
            os.remove(file_path)
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

    def _process_prediction(self, prediction: dict) -> dict:
        """Process prediction.

        Args:
            prediction (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - basename (str): The image name.
                - masks (numpy.ndarray): Shape (N, H, W), the predicted masks.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
                - mask_scores (np.array, optional): Shape (N, ), the predicted
                  scores of masks.

        Returns:
            dict: The processed prediction results. With key `pred_txt`.
        """
        if self.classes is None:
            self._get_classes()
        pred = dict()
        basename = prediction['basename']

        pred_txt = osp.join(self.outfile_prefix, basename + '_pred.txt')
        pred['pred_txt'] = pred_txt

        labels = prediction['labels']
        masks = prediction['masks']
        mask_scores = prediction['mask_scores']
        with open(pred_txt, 'w') as f:
            for i, (label, mask,
                    mask_score) in enumerate(zip(labels, masks, mask_scores)):
                class_name = self.classes[label]  # type: ignore
                class_id = CSLabels.name2label[class_name].id
                png_filename = osp.join(self.outfile_prefix,
                                        basename + f'_{i}_{class_name}.png')
                imwrite(mask, png_filename)
                f.write(f'{osp.basename(png_filename)} '
                        f'{class_id} {mask_score}\n')
        return pred

    def _get_classes(self) -> None:
        """Get classes from self.dataset_meta."""
        if self.dataset_meta and 'classes' in self.dataset_meta:
            self.classes = self.dataset_meta['classes']
        elif self.dataset_meta and 'CLASSES' in self.dataset_meta:
            self.classes = self.dataset_meta['CLASSES']
            warnings.warn(
                'DeprecationWarning: The `CLASSES` in `dataset_meta` is '
                'deprecated, use `classes` instead!')
        else:
            raise RuntimeError('Could not find `classes` in dataset_meta: '
                               f'{self.dataset_meta}')
