# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import multiprocessing
import numpy as np
import os
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from logging import Logger
from terminaltables import AsciiTable
from typing import Dict, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import get_local_path

try:
    from panopticapi.evaluation import OFFSET, VOID, PQStat
    from panopticapi.utils import id2rgb, rgb2id

    from mmeval.metrics.utils.coco_wrapper import COCOPanoptic as PanopticAPI
    HAS_PANOPTICAPI = True
except ImportError:
    HAS_PANOPTICAPI = False

try:
    from mmcv import imread, imwrite
except ImportError:
    from mmeval.utils import imread, imwrite

default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)

# A custom value to distinguish instance ID and category ID; need to
# be greater than the number of categories.
# For a pixel in the panoptic result map:
#   pan_id = ins_id * INSTANCE_OFFSET + cat_id
INSTANCE_OFFSET = 1000


class CocoPanoptic(BaseMetric):
    """COCO panoptic segmentation evaluation metric.

    Evaluate PQ, SQ RQ for panoptic segmentation tasks. Please refer to
    https://cocodataset.org/#panoptic-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to return the computed  results of each
            class. Defaults to False.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When ``nproc`` exceeds the number of cpu cores,
            the number of cpu cores is used.
        logger (Logger, optional): logger used to record messages. When set to
            ``None``, the default logger will be used.
            Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 nproc: int = 32,
                 logger: Optional[Logger] = None,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        if not HAS_PANOPTICAPI:
            raise RuntimeError('Failed to import `panopticapi`. Please '
                               'try to install official pycocotools by '
                               '"pip install git+https://github.com/'
                               'cocodataset/panopticapi.git"')
        super().__init__(**kwargs)

        self.tmp_dir = None
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

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

        self.nproc = nproc
        self.seg_prefix = seg_prefix
        self.classwise = classwise

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        self._coco_api: Optional[PanopticAPI]  # type: ignore

        if ann_file is not None:
            with get_local_path(
                    filepath=ann_file,
                    backend_args=backend_args) as local_path:
                self._coco_api = PanopticAPI(annotation_file=local_path)
        else:
            self._coco_api = None

        self.backend_args = backend_args
        # TODO: update after logger is supported in BaseMetric
        self.logger = default_logger if logger is None else logger

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - image_id (int): Image id.
                - sem_seg (numpy.ndarray): The prediction of semantic
                  segmentation.
                - segm_file (str): Segmentation file name.

            groundtruths (Sequence[dict]): A sequence of dict. If load from
                `ann_file`, the dict inside can be empty. Else, each dict
                represents a groundtruths for an image, with the following
                keys:

                - image_id (int): Image id.
                - segm_file (str): Segmentation file name.
                - width (int): The width of the image.
                - height (int): The height of the image.
                - segments_info (list, optional): groundtruth information
                  list. Is necessary if self._coco_api is None.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'  # noqa: E501
            prediction = self._process_prediction(prediction)

            if self.tmp_dir is None:
                # add the prediction and groundtruth into `self._results`, and
                # compute the results after all images have been inferenced.
                self._results.append((prediction, groundtruth))
            else:
                # Directly compute the results and put into `self._results`.
                pq_results = self._compute_single_pq_stats(
                    prediction, groundtruth)
                self._results.append(pq_results)

    def compute_metric(self, results: list) -> dict:
        """Compute the CocoPanoptic metric.

        Args:
            results (List[tuple] | List[PQStat]): If self.tmp_dir is
                None, it is a list of PQStat, else a list of tuple, each
                tuple is the prediction and ground truth of an image.
                The list has already been synced across all ranks.

        Returns:
            dict: The computed Panoptic Quality metric.
        """
        eval_results: OrderedDict = OrderedDict()

        if self.format_only:
            self.logger.info('Results are saved in '    # type: ignore
                             f'{osp.dirname(self.outfile_prefix)}')  # type: ignore # yapf: disable # noqa: E501
        if self.tmp_dir:
            pq_stat_results = results
        else:
            pq_stat_results = self._compute_multi_pq_stats(results)

        # aggregate the results generated in process
        pq_stat = PQStat()
        for result in pq_stat_results:
            pq_stat += result

        categories = self.categories
        classes = self.classes

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = dict()
        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                categories, isthing=isthing)
            if name == 'All' and self.classwise:
                # avoid classes is not same as categories.
                if len(classes) < len(classwise_results):
                    for category in categories.values():
                        if category['name'] not in classes:
                            class_id = category['id']
                            classwise_results.pop(class_id)
                pq_results['classwise'] = classwise_results

        # print tables
        self._print_panoptic_table(pq_results)

        # process results
        eval_results['PQ'] = pq_results['All']['pq']
        eval_results['SQ'] = pq_results['All']['sq']
        eval_results['RQ'] = pq_results['All']['rq']
        eval_results['PQ_th'] = pq_results['Things']['pq']
        eval_results['SQ_th'] = pq_results['Things']['sq']
        eval_results['RQ_th'] = pq_results['Things']['rq']
        eval_results['PQ_st'] = pq_results['Stuff']['pq']
        eval_results['SQ_st'] = pq_results['Stuff']['sq']
        eval_results['RQ_st'] = pq_results['Stuff']['rq']
        classwise_results = pq_results.get('classwise')
        if classwise_results is not None:
            for k, v in zip(classes, classwise_results.values()):
                eval_results[f'{k}_PQ'] = v['pq']

        return eval_results

    def __del__(self) -> None:
        """Clean up the results if necessary."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def _process_prediction(self, prediction: dict) -> dict:
        """Process panoptic segmentation predictions.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - image_id (int): Image id.
                - sem_seg (numpy.ndarray): The prediction of semantic
                  segmentation.
                - segm_file (str): Segmentation file name.

        Returns:
            dict: The processed predictions.
        """
        classes = self.classes
        if self._coco_api is not None:
            label2cat = self.label2cat
        else:
            label2cat = None  # type: ignore # noqa: E501

        pan = prediction.pop('sem_seg', None)
        assert pan is not None

        pan_labels = np.unique(pan)
        segments_info = []

        for pan_label in pan_labels:
            sem_label = pan_label % INSTANCE_OFFSET
            # We reserve the length of classes for VOID label
            if sem_label == len(classes):
                continue
            mask = pan == pan_label
            area = mask.sum()
            # when ann_file provided, sem_label should be cat_id, otherwise
            # sem_label should be a continuous id, not the cat_id
            # defined in dataset
            category_id = label2cat[sem_label] if label2cat else sem_label
            segments_info.append({
                'id': int(pan_label),
                'category_id': category_id,
                'area': int(area)
            })

        # evaluation script uses 0 for VOID label.
        pan[pan % INSTANCE_OFFSET == len(classes)] = VOID
        pan = id2rgb(pan).astype(np.uint8)

        segm_file = osp.join(self.outfile_prefix, prediction['segm_file'])  # type: ignore # yapf: disable # noqa: E501
        imwrite(pan[:, :, ::-1], segm_file)
        prediction['segments_info'] = segments_info
        return prediction

    @staticmethod
    def _compute_pq_stats(prediction: dict,
                          groundtruth: dict,
                          categories: dict,
                          seg_prefix: str,
                          outfile_prefix: str,
                          backend_args: Optional[dict] = None,
                          coco_api: Optional[PanopticAPI] = None) -> PQStat:
        """The function to compute the metric of Panoptic Segmentation.

        Args:
            prediction (dict): Same as :class:`CocoPanoptic.add`.
            groundtruth (dict): Same as :class:`CocoPanoptic.add`.
            categories (dict): The categories of the dataset.
            seg_prefix (str): Same as :class:`CocoPanoptic.seg_prefix`.
            outfile_prefix (str): Same as :class:`CocoPanoptic.outfile_prefix`.
            backend_args (dict, optional): Same as
                :class:`CocoPanoptic.backend_args`.
            coco_api (PanopticAPI, optional): PanopticAPI wrapper.
                Defaults to None.

        Returns:
            PQStat: The metric results of Panoptic Segmentation.
        """
        # get panoptic groundtruth and prediction
        gt_seg_map_path = osp.join(seg_prefix, groundtruth['segm_file'])
        pred_seg_map_path = osp.join(outfile_prefix, prediction['segm_file'])

        pan_gt = imread(
            gt_seg_map_path,
            flag='color',
            channel_order='rgb',
            backend_args=backend_args)
        pan_pred = imread(pred_seg_map_path, flag='color', channel_order='rgb')

        if coco_api is None:
            pan_png = pan_gt.squeeze()
            pan_png = rgb2id(pan_png)

            gt_segments_info = []
            for segment_info in groundtruth['segments_info']:
                id = segment_info['id']
                label = segment_info['category']
                mask = pan_png == id
                isthing = categories[label]['isthing']
                if isthing:
                    iscrowd = 1 if not segment_info['is_thing'] else 0
                else:
                    iscrowd = 0

                new_segment_info = {
                    'id': id,
                    'category_id': label,
                    'isthing': isthing,
                    'iscrowd': iscrowd,
                    'area': mask.sum()
                }
                gt_segments_info.append(new_segment_info)
        else:
            # get segments_info from annotation file
            gt_segments_info = coco_api.imgToAnns[groundtruth['image_id']]

        # process pq
        pq_stat = PQStat()
        pan_gt = rgb2id(pan_gt)
        pan_pred = rgb2id(pan_pred)
        gt_segms = {el['id']: el for el in gt_segments_info}
        pred_segms = {el['id']: el for el in prediction['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = {el['id'] for el in prediction['segments_info']}
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    'In the image with ID {} segment with ID {} is '
                    'presented in PNG and not presented in JSON.'.format(
                        groundtruth['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError(
                    'In the image with ID {} segment with ID {} has '
                    'unknown category_id {}.'.format(
                        groundtruth['image_id'], label,
                        pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError(
                'In the image with ID {} the following segment IDs {} '
                'are presented in JSON and not presented in PNG.'.format(
                    groundtruth['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(
            np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                    'category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                'area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info['category_id']], pred_label),
                    0)
            # predicted segment is ignored if more than half of
            # the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1

        return pq_stat

    def _compute_single_pq_stats(self, prediction: dict,
                                 groundtruth: dict) -> PQStat:
        """Compute single image of the Panoptic Segmentation metric. Used when
        `self.tmp_dir` is None.

        Args:
            prediction (dict): Same as :class:`CocoPanoptic.add`.
            groundtruth (dict): Same as :class:`CocoPanoptic.add`.

        Returns:
            PQStat: The metric results of Panoptic Segmentation.
        """
        categories = self.categories

        pq_stat = self._compute_pq_stats(
            prediction=prediction,
            groundtruth=groundtruth,
            categories=categories,
            seg_prefix=self.seg_prefix,  # type: ignore
            outfile_prefix=self.outfile_prefix,  # type: ignore
            backend_args=self.backend_args,
            coco_api=self._coco_api)

        return pq_stat

    def _compute_multi_pq_stats(self, results: list) -> list:
        """Compute multi images of the Panoptic Segmentation metric. Used when
        self.direct_compute is False.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            List[PQStat]: A list of the metric results of Panoptic
            Segmentation.
        """
        predictions, groundtruths = zip(*results)
        num_images = len(predictions)

        categories = self.categories

        nproc = min(self.nproc, multiprocessing.cpu_count())
        if nproc > 1:
            pool = multiprocessing.Pool(nproc)
            pq_stat_results = pool.starmap(
                self._compute_pq_stats,
                zip(predictions, groundtruths, [categories] * num_images,
                    [self.seg_prefix] * num_images,
                    [self.outfile_prefix] * num_images,
                    [self.backend_args] * num_images,
                    [self._coco_api] * num_images))
            pool.close()
        else:
            pq_stat_results = []
            for img_idx in range(num_images):
                pq_stat = self._compute_pq_stats(
                    prediction=predictions[img_idx],
                    groundtruth=groundtruths[img_idx],
                    categories=categories,
                    seg_prefix=self.seg_prefix,  # type: ignore
                    outfile_prefix=self.outfile_prefix,  # type: ignore
                    backend_args=self.backend_args,
                    coco_api=self._coco_api)
                pq_stat_results.append(pq_stat)
        return pq_stat_results

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

    @property
    def thing_classes(self) -> tuple:
        """Get thing classes from self.dataset_meta."""
        if self.dataset_meta and 'thing_classes' in self.dataset_meta:
            thing_classes = self.dataset_meta['thing_classes']
        elif self.dataset_meta and 'THING_CLASSES' in self.dataset_meta:
            thing_classes = self.dataset_meta['THING_CLASSES']
            warnings.warn(
                'DeprecationWarning: The `THING_CLASSES` in `dataset_meta` '
                'is deprecated, use `thing_classes` instead!')
        else:
            raise RuntimeError('Could not find `thing_classes` in '
                               f'dataset_meta: {self.dataset_meta}')
        return thing_classes

    @property
    def label2cat(self) -> dict:
        """Get `label2cat` from `self._coco_api`."""
        assert self._coco_api is not None
        classes = self.classes
        cat_ids = self._coco_api.get_cat_ids(cat_names=classes)  # type: ignore
        label2cat = {i: cat_id
                     for i, cat_id in enumerate(cat_ids)}  # type: ignore
        return label2cat

    @property
    def categories(self) -> dict:
        """Get `categories`."""
        if self._coco_api is None:
            classes = self.classes
            thing_classes = self.thing_classes
            categories = dict()
            for id, name in enumerate(classes):
                isthing = 1 if name in thing_classes else 0
                categories[id] = {'id': id, 'name': name, 'isthing': isthing}
        else:
            categories = self._coco_api.cats
        return categories

    def _print_panoptic_table(self, pq_results: dict) -> None:
        """Print the panoptic evaluation results table.

        Args:
            pq_results: The Panoptic Quality results.
        """
        table_title = ' Panoptic Results'
        headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
        data = [headers]
        for name in ['All', 'Things', 'Stuff']:
            numbers = [
                f'{round(pq_results[name][k] * 100, 3):0.3f}'
                for k in ['pq', 'sq', 'rq']
            ]
            row = [name] + numbers + [pq_results[name]['n']]
            data.append(row)
        table = AsciiTable(data, title=table_title)
        self.logger.info(f'Panoptic Evaluation Results:\n {table.table}')

        if self.classwise:
            classwise_table_title = ' Classsiwe Panoptic Results'
            classwise_results = pq_results.get('classwise', None)
            classes = self.classes

            assert classwise_results is not None
            assert len(classwise_results) == len(classes)
            class_metrics = [
                (classes[i], ) + tuple(f'{round(metrics[k] * 100, 3):0.3f}'
                                       for k in ['pq', 'sq', 'rq'])
                for i, metrics in enumerate(classwise_results.values())
            ]
            num_columns = min(8, len(class_metrics) * 4)
            results_flatten = list(itertools.chain(*class_metrics))
            headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            data = [headers]
            data += [result for result in results_2d]
            table = AsciiTable(data, title=classwise_table_title)
            self.logger.info(
                f'Classwise Panoptic Evaluation Results:\n {table.table}')
