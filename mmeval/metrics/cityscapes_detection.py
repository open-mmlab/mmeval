# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from terminaltables import AsciiTable
from typing import Dict, Optional, Sequence, Union

from mmeval.core.base_metric import BaseMetric

try:
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa: E501
    import cityscapesscripts.helpers.labels as CSLabels

    from .utils.cityscapes_wrapper import evaluateImgLists
    HAS_CITYSCAPESAPI = True
except ImportError:
    HAS_CITYSCAPESAPI = False

try:
    from mmcv import imwrite
except ImportError:
    from mmeval.utils import imwrite


class CityScapesDetection(BaseMetric):
    """CityScapes metric for instance segmentation.

    Args:
        outfile_prefix (str): The prefix of txt and png files. It is the
            saving path of txt and png file, e.g. "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            cityscapes instance segmentation masks. It's necessary when
            training and validation. It could be None when infer on test
            dataset. Defaults to None.
        format_only (bool): Format the output results without performing
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        classwise (bool): Whether to return the computed  results of each
            class. Defaults to False.
        dump_matches (bool): Whether dump matches.json file during evaluating.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
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
        ...     seg_prefix=seg_prefix,
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
        ...     prediction = {
        ...         'basename': basename1,
        ...         'mask_scores': np.array([1.0]),
        ...         'labels': np.array([0]),
        ...         'masks': dummy_mask1
        ...     }
        ...     groundtruth = {
        ...         'file_name': img_path1
        ...     }
        ...
        ...     return [prediction], [groundtruth]
        >>>
        >>> predictions, groundtruths = _gen_fake_datasamples(seg_prefix)
        >>> cityscapes_det_metric(predictions, groundtruths)  # doctest: +ELLIPSIS  # noqa: E501
        {'mAP': ..., 'AP50': ...}
        >>> tmp_dir.cleanup()
    """

    def __init__(self,
                 outfile_prefix: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 format_only: bool = False,
                 classwise: bool = False,
                 dump_matches: bool = False,
                 backend_args: Optional[dict] = None,
                 **kwargs):

        if not HAS_CITYSCAPESAPI:
            raise RuntimeError('Failed to import `cityscapesscripts`.'
                               'Please try to install official '
                               'cityscapesscripts by '
                               '"pip install cityscapesscripts"')
        super().__init__(**kwargs)

        self.tmp_dir = None
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        else:
            assert seg_prefix is not None, '`seg_prefix` is necessary when '
            'computing the CityScapes metrics'

        # outfile_prefix should be a prefix of a path which points to a shared
        # storage when train or test with multi nodes.
        self.outfile_prefix = outfile_prefix
        if outfile_prefix is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(self.tmp_dir.name, 'results')
        else:
            # the directory to save predicted panoptic segmentation mask
            self.outfile_prefix = osp.join(self.outfile_prefix, 'results')  # type: ignore # yapf: disable # noqa: E501
        # make dir to avoid potential error
        dir_name = osp.expanduser(self.outfile_prefix)
        os.makedirs(dir_name, exist_ok=True)

        self.seg_prefix = seg_prefix
        self.classwise = classwise
        self.backend_args = backend_args
        self.dump_matches = dump_matches

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
            self.logger.info('Results are saved in '    # type: ignore
                             f'{osp.dirname(self.outfile_prefix)}')  # type: ignore # yapf: disable # noqa: E501
            return eval_results

        gt_instances_file = osp.join(self.outfile_prefix, 'gtInstances.json')  # type: ignore # yapf: disable # noqa: E501
        # split gt and prediction list
        preds, gts = zip(*results)
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = gt_instances_file

        groundtruth_list = [gt['file_name'] for gt in gts]
        prediction_list = [pred['pred_txt'] for pred in preds]
        CSEval_results = evaluateImgLists(
            prediction_list,
            groundtruth_list,
            CSEval.args,
            self.backend_args,
            dump_matches=self.dump_matches)['averages']

        map = float(CSEval_results['allAp'])
        map_50 = float(CSEval_results['allAp50%'])

        eval_results['mAP'] = map
        eval_results['AP50'] = map_50

        results_list = ('mAP', f'{round(map * 100, 2):0.2f}',
                        f'{round(map_50 * 100, 2):0.2f}')

        if self.classwise:
            results_per_category = []
            for category, aps in CSEval_results['classes'].items():
                eval_results[f'{category}_ap'] = float(aps['ap'])
                eval_results[f'{category}_ap50'] = float(aps['ap50%'])
                results_per_category.append(
                    (f'{category}', f'{round(float(aps["ap"]) * 100, 2):0.2f}',
                     f'{round(float(aps["ap50%"]) * 100, 2):0.2f}'))
            results_per_category.append(results_list)
            eval_results['results_list'] = results_per_category
        else:
            eval_results['results_list'] = [results_list]

        self._print_panoptic_table(eval_results)
        return eval_results

    def __del__(self) -> None:
        """Clean up the results if necessary."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

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
        classes = self.classes
        pred = dict()
        basename = prediction['basename']

        pred_txt = osp.join(self.outfile_prefix, basename + '_pred.txt')  # type: ignore # yapf: disable # noqa: E501
        pred['pred_txt'] = pred_txt

        labels = prediction['labels']
        masks = prediction['masks']
        mask_scores = prediction['mask_scores']
        with open(pred_txt, 'w') as f:
            for i, (label, mask,
                    mask_score) in enumerate(zip(labels, masks, mask_scores)):
                class_name = classes[label]
                class_id = CSLabels.name2label[class_name].id
                png_filename = osp.join(
                    self.outfile_prefix, basename + f'_{i}_{class_name}.png')  # type: ignore # yapf: disable # noqa: E501
                imwrite(mask, png_filename)
                f.write(f'{osp.basename(png_filename)} '
                        f'{class_id} {mask_score}\n')
        return pred

    @property
    def classes(self) -> tuple:
        """Get classes from self.dataset_meta."""
        if self.dataset_meta and 'classes' in self.dataset_meta:
            classes = self.dataset_meta['classes']
        elif self.dataset_meta and 'CLASSES' in self.dataset_meta:
            classes = self.dataset_meta['CLASSES']
            warnings.warn(
                'DeprecationWarning: The `CLASSES` in `dataset_meta` is '
                'deprecated, use `classes` instead!')
        else:
            raise RuntimeError('Could not find `classes` in dataset_meta: '
                               f'{self.dataset_meta}')
        return classes

    def _print_panoptic_table(self, eval_results: dict) -> None:
        """Print the evaluation results table.

        Args:
            eval_results (dict): The computed metric.
        """
        result = eval_results.pop('results_list')

        header = ['class', 'AP(%)', 'AP50(%)']
        table_title = ' Cityscapes Results'

        results_flatten = list(itertools.chain(*result))

        results_2d = itertools.zip_longest(
            *[results_flatten[i::3] for i in range(3)])
        table_data = [header]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data, title=table_title)
        table.inner_footing_row_border = True
        self.logger.info(f'CityScapes Evaluation Results: \n {table.table}')
