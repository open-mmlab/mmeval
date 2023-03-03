# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import multiprocessing
import numpy as np
import os
import os.path as osp
import shutil
import tempfile
import warnings
from collections import OrderedDict
from logging import Logger
from terminaltables import AsciiTable
from typing import Dict, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import get, get_local_path
from .utils import imread, imwrite

try:
    from panopticapi.evaluation import OFFSET, VOID, PQStat
    from panopticapi.utils import id2rgb, rgb2id

    from mmeval.metrics.utils.coco_wrapper import COCOPanoptic as PanopticAPI
    HAS_PANOPTICAPI = True
except ImportError:
    HAS_PANOPTICAPI = False

try:
    import mmcv
    HAS_MMCV = True
except ImportError:
    HAS_MMCV = False

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
        keep_results (bool): Whether to keep the results. When ``format_only``
            is True, ``keep_results`` must be True. If False, the result files
            will remove after compute the metric. Defaults to False.
        direct_compute (bool): Whether to compute metric on each inference
            iteration. Defaults to True.
        nproc (int): Number of processes for panoptic quality computing. It
            will be used when `direct_compute` is False. Defaults to 32.
            When ``nproc`` exceeds the number of cpu cores, the number of
            cpu cores is used.
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
                 keep_results: bool = False,
                 direct_compute: bool = True,
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
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
            assert keep_results, '`keep_results` must be `True` when '
            'format_only is True, otherwise the result files will be cleaned '
            'up at the end.'

        self.tmp_dir = None
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
            self.categories = self._coco_api.cats
        else:
            self._coco_api = None
            self.categories = None

        self.backend_args = backend_args
        self.keep_results = keep_results
        self.direct_compute = direct_compute
        # TODO: update after logger is supported in BaseMetric
        self.logger = default_logger if logger is None else logger

        # necessary arguments, which can be obtained from self.dataset_meta.
        # Note that they cannot be force set during initialization, because
        # self.dataset_meta may be updated after build Metric.
        self.label2cat = None
        self.classes = None
        self.thing_classes = None

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

            if self.direct_compute:
                # Directly compute the results and put into `self._results`
                pq_results = self._compute_single_pq_stats(
                    prediction, groundtruth)
                self._results.append(pq_results)
            else:
                self._results.append((prediction, groundtruth))

    def compute_metric(self, results: list) -> dict:
        """Compute the CocoPanoptic metric.

        Args:
            results (List[tuple] | List[PQStat]): If self.direct_compute is
                True, it is a list of PQStat, else a list of tuple, each
                tuple is the prediction and ground truth of an image.
                The list has already been synced across all ranks.

        Returns:
            dict: The computed Panoptic Quality metric.
        """
        eval_results: OrderedDict = OrderedDict()

        if self.format_only:
            self.logger.info(f'Results are saved in {osp.dirname(self.outfile_prefix)}')  # type: ignore # yapf: disable # noqa: E501
            return eval_results

        if self.classes is None:
            self._get_classes()

        if self.direct_compute:
            pq_stat_results = results
        else:
            pq_stat_results = self._compute_multi_pq_stats(
                results)  # type: ignore

        # aggregate the results generated in process
        pq_stat = PQStat()
        for result in pq_stat_results:
            pq_stat += result

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = dict()
        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All' and self.classwise:
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
            for k, v in zip(self.classes, classwise_results.values()):  # type: ignore # yapf: disable # noqa: E501
                eval_results[f'{k}_PQ'] = v['pq']

        # remove result files to save storage space and make dir again
        # to avoid potential error
        if not self.keep_results:
            if osp.exists(self.outfile_prefix):  # type: ignore # yapf: disable # noqa: E501
                shutil.rmtree(self.outfile_prefix)  # type: ignore # yapf: disable # noqa: E501
                dir_name = osp.expanduser(self.outfile_prefix)  # type: ignore # yapf: disable # noqa: E501
                os.makedirs(dir_name, exist_ok=True)  # type: ignore # yapf: disable # noqa: E501
        return eval_results

    def __del__(self) -> None:
        """Clean up the results if necessary."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
        elif (not self.dist_comm.is_initialized or self.dist_comm.world_size
              == 1 or self.dist_comm.rank == 0) and not self.keep_results:
            shutil.rmtree(self.outfile_prefix)  # type: ignore

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
        if self.classes is None:
            self._get_classes()

        if self.label2cat is None and self._coco_api is not None:
            self._get_label2cat()

        pan = prediction.pop('sem_seg', None)
        assert pan is not None

        pan_labels = np.unique(pan)
        segments_info = []

        for pan_label in pan_labels:
            sem_label = pan_label % INSTANCE_OFFSET
            # We reserve the length of self.classes for VOID label
            if sem_label == len(self.classes):  # type: ignore # yapf: disable # noqa: E501
                continue
            mask = pan == pan_label
            area = mask.sum()
            # when ann_file provided, sem_label should be cat_id, otherwise
            # sem_label should be a continuous id, not the cat_id
            # defined in dataset
            category_id = self.label2cat[sem_label] if self.label2cat else sem_label   # type: ignore # yapf: disable # noqa: E501
            segments_info.append({
                'id': int(pan_label),
                'category_id': category_id,
                'area': int(area)
            })

        # evaluation script uses 0 for VOID label.
        pan[pan % INSTANCE_OFFSET == len(self.classes)] = VOID  # type: ignore # yapf: disable # noqa: E501
        pan = id2rgb(pan).astype(np.uint8)

        segm_file = osp.join(self.outfile_prefix, prediction['segm_file'])  # type: ignore # yapf: disable # noqa: E501
        if HAS_MMCV:
            mmcv.imwrite(pan[:, :, ::-1], segm_file)
        else:
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

        if HAS_MMCV:
            img_bytes = get(gt_seg_map_path, backend_args=backend_args)
            pan_gt = mmcv.imfrombytes(
                img_bytes, flag='color', channel_order='rgb')
            img_bytes = get(pred_seg_map_path, backend_args=backend_args)
            pan_pred = mmcv.imfrombytes(
                img_bytes, flag='color', channel_order='rgb')
        else:
            pan_gt = imread(
                gt_seg_map_path,
                backend_args=backend_args,
                flag='color',
                channel_order='rgb')
            pan_pred = imread(
                pred_seg_map_path,
                backend_args=backend_args,
                flag='color',
                channel_order='rgb')

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
        self.direct_compute is True.

        Args:
            prediction (dict): Same as :class:`CocoPanoptic.add`.
            groundtruth (dict): Same as :class:`CocoPanoptic.add`.

        Returns:
            PQStat: The metric results of Panoptic Segmentation.
        """
        if self.classes is None:
            self._get_classes()
        if self.thing_classes is None:
            self._get_thing_classes()
        if self.categories is None:
            self._get_categories()

        pq_stat = self._compute_pq_stats(
            prediction=prediction,
            groundtruth=groundtruth,
            categories=self.categories,  # type: ignore
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

        nproc = min(self.nproc, multiprocessing.cpu_count())
        if nproc > 1:
            pool = multiprocessing.Pool(nproc)
            pq_stat_results = pool.starmap(
                self._compute_pq_stats,
                zip(predictions, groundtruths, [self.categories] * num_images,
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
                    categories=self.categories,  # type: ignore
                    seg_prefix=self.seg_prefix,  # type: ignore
                    outfile_prefix=self.outfile_prefix,  # type: ignore
                    backend_args=self.backend_args,
                    coco_api=self._coco_api)
                pq_stat_results.append(pq_stat)
        return pq_stat_results

    def _get_label2cat(self) -> None:
        """Get `label2cat` from `self._coco_api`."""
        cat_ids = self._coco_api.get_cat_ids(cat_names=self.classes)  # type: ignore # yapf: disable # noqa: E501
        self.label2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}  # type: ignore # yapf: disable # noqa: E501

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

    def _get_thing_classes(self) -> None:
        """Get classes from self.dataset_meta."""
        if self.dataset_meta and 'thing_classes' in self.dataset_meta:
            self.thing_classes = self.dataset_meta['thing_classes']
        elif self.dataset_meta and 'THING_CLASSES' in self.dataset_meta:
            self.thing_classes = self.dataset_meta['THING_CLASSES']
            warnings.warn(
                'DeprecationWarning: The `THING_CLASSES` in `dataset_meta` '
                'is deprecated, use `thing_classes` instead!')
        else:
            raise RuntimeError('Could not find `thing_classes` in '
                               f'dataset_meta: {self.dataset_meta}')

    def _get_categories(self) -> None:
        """Get dataset categories."""
        if self._coco_api is None:
            if self.classes is None:
                self._get_classes()
            if self.thing_classes is None:
                self._get_thing_classes()
            categories = dict()
            for id, name in enumerate(self.classes):  # type: ignore # yapf: disable # noqa: E501
                isthing = 1 if name in self.thing_classes else 0  # type: ignore # yapf: disable # noqa: E501
                categories[id] = {'id': id, 'name': name, 'isthing': isthing}
        else:
            categories = self._coco_api.cats
        self.categories = categories

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
            assert classwise_results is not None
            assert len(classwise_results) == len(self.classes)  # type: ignore
            class_metrics = [
                (self.classes[i], ) +  # type: ignore
                tuple(f'{round(metrics[k] * 100, 3):0.3f}'
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
