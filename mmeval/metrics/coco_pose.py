# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import logging
import numpy as np
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from json import dump
from typing import Dict, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import load
from .utils import oks_nms, soft_oks_nms

try:
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval
    HAS_XTCOCOTOOLS = True
except ImportError:
    HAS_XTCOCOTOOLS = False

logger = logging.getLogger(__name__)


class COCOKeyPointDetection(BaseMetric):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to True.
        iou_type (str): The same parameter as ``iouType`` in
            :class:`xtcocotools.COCOeval`, which
            should be one of the following options:

            - ``'keypoints'``.
            - ``'keypoints_crowd'`` (used in CrowdPose dataset).

            Defaults to ``'keypoints'``.
        score_mode (str): The mode to score the prediction results, which
            should be one of the following options:

            - ``'bbox'``: Take the score of bbox as the score of the
              prediction results.
            - ``'bbox_keypoint'``: Use keypoint score to rescore the
              prediction results.
            - ``'bbox_rle'``: Use rle_score to rescore the
              prediction results.

            Defaults to ``'bbox_keypoint'``.
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``'bbox_keypoint'``. Defaults to 0.2.
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

            - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
              perform NMS.
            - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
              to perform soft NMS.
            - ``'none'``: Do not perform NMS. Typically for bottomup mode
              output.

            Defaults to ``'oks_nms'``.
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to 0.9.
        format_only (bool): Whether only formats the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to False.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created.
            Defaults to None.
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`. Must
            include key ``'dataset_meta'`` in order to compute the metric.

    Examples:
        >>> import copy
        >>> import numpy as np
        >>> import tempfile
        >>> from mmeval.fileio import load
        >>> from mmeval.metrics import COCOKeyPointDetection
        >>> try:
        ...     from xtcocotools.coco import COCO
        ...     from xtcocotools.cocoeval import COCOeval
        ...     HAS_XTCOCOTOOLS = True
        ... except ImportError:
        ...     HAS_XTCOCOTOOLS = False
        >>> ann_file = 'tests/test_metrics/data/coco_pose_sample.json'
        >>> coco = COCO(ann_file)
        loading annotations into memory...
        Done (t=0.00s)
        creating index...
        index created!
        >>> classes = coco.loadCats(coco.getCatIds())
        >>> coco_dataset_meta = {
        ...     'CLASSES': 'classes',
        ...     'num_keypoints': 17,
        ...     'sigmas': None,
        ... }
        >>> predictions, groundtruths = [], []
        >>> db = load(ann_file)
        >>> for ann in db['annotations']:
        ...     bboxes = np.array(
        ...         ann['bbox'], dtype=np.float32).reshape(-1, 4)
        ...     keypoints = np.array(ann['keypoints']).reshape((1, -1, 3))
        ...     prediction = {
        ...         'id': ann['id'],
        ...         'img_id': ann['image_id'],
        ...         'bboxes': bboxes,
        ...         'keypoints': keypoints[..., :2],
        ...         'keypoint_scores': keypoints[..., -1],
        ...         'bbox_scores': np.ones((1, ), dtype=np.float32),
        ...         'area': ann['area'],
        ...     }
        ...     groundtruth = {
        ...         'img_id': ann['image_id'],
        ...         'width': 640,
        ...         'height': 480,
        ...         'num_keypoints': ann['num_keypoints'],
        ...         'raw_ann_info': [copy.deepcopy(ann)],
        ...     }
        ...     predictions.append(prediction)
        ...     groundtruths.append(groundtruth)
        >>> coco_pose_metric = COCOKeyPointDetection(
        ...     ann_file=ann_file,
        ...     dataset_meta=coco_dataset_meta)
        loading annotations into memory...
        Done (t=0.00s)
        creating index...
        index created!
        >>> coco_pose_metric(predictions,
        ...                  [dict() for _ in range(len(predictions))])
        Loading and preparing results...
        DONE (t=0.00s)
        creating index...
        index created!
        Running per image evaluation...
        Evaluate annotation type *keypoints*
        DONE (t=0.00s).
        Accumulating evaluation results...
        DONE (t=0.00s).
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  1.000  # noqa: E501
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  1.000
        OrderedDict([('AP', 1.0), ('AP .5', 1.0), ('AP .75', 1.0),
        ('AP (M)', 1.0), ('AP (L)', 1.0), ('AR', 1.0), ('AR .5', 1.0),
        ('AR .75', 1.0), ('AR (M)', 1.0), ('AR (L)', 1.0)])
        >>> coco_pose_metric = COCOKeyPointDetection(dataset_meta=coco_dataset_meta)
        >>> coco_pose_metric.add(predictions, groundtruths)
        >>> coco_pose_metric.compute()
        Loading and preparing results...
        DONE (t=0.00s)
        creating index...
        index created!
        Running per image evaluation...
        Evaluate annotation type *keypoints*
        DONE (t=0.00s).
        Accumulating evaluation results...
        DONE (t=0.00s).
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  1.000  # noqa: E501
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  1.000
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  1.000
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  1.000
        OrderedDict([('AP', 1.0), ('AP .5', 1.0), ('AP .75', 1.0),
        ('AP (M)', 1.0), ('AP (L)', 1.0), ('AR', 1.0), ('AR .5', 1.0),
        ('AR .75', 1.0), ('AR (M)', 1.0), ('AR (L)', 1.0)])
    """

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 **kwargs) -> None:
        if not HAS_XTCOCOTOOLS:
            raise RuntimeError(
                'Failed to import `COCO` and `COCOeval` from xtcocotools.'
                'Please try to install xtcocotools by '
                '`pip install xtcocotools`')
        super().__init__(**kwargs)
        self.dataset_meta: dict
        self.ann_file = ann_file
        # initialize coco helper with the annotation json file
        # if ann_file is not specified, initialize with the converted dataset
        if ann_file is not None:
            self.coco = COCO(ann_file)
        else:
            self.coco = None

        self.use_area = use_area
        self.iou_type = iou_type

        allowed_score_modes = ['bbox', 'bbox_keypoint', 'bbox_rle']
        if score_mode not in allowed_score_modes:
            raise ValueError(
                "`score_mode` should be one of 'bbox', 'bbox_keypoint', "
                f"'bbox_rle', but got {score_mode}")
        self.score_mode = score_mode
        self.keypoint_score_thr = keypoint_score_thr

        allowed_nms_modes = ['oks_nms', 'soft_oks_nms', 'none']
        if nms_mode not in allowed_nms_modes:
            raise ValueError(
                "`nms_mode` should be one of 'oks_nms', 'soft_oks_nms', "
                f"'none', but got {nms_mode}")
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr

        if format_only:
            assert outfile_prefix is not None, '`outfile_prefix` can not be '\
                'None when `format_only` is True, otherwise the result file '\
                'will be saved to a temp directory which will be cleaned up '\
                'in the end.'
        elif ann_file is not None:
            # do evaluation only if the ground truth annotations exist
            assert 'annotations' in load(ann_file), \
                'Ground truth annotations are required for evaluation '\
                'when `format_only` is False.'

        self.format_only = format_only
        self.outfile_prefix = outfile_prefix

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a pose estimation result for an instance, with
                the following keys:

                - id (int): the id of the instance
                - img_id (int): the image_id of the instance
                - bboxes: the bounding bboxes of the instance
                - keypoints (np.ndarray): the predicted keypoints coordinates
                  of the instance with shape (N, K, 2), where N is the number
                  of instances and K is the number of keypoints. For
                  topdown-style output, N is usually 1, while for
                  bottomup-style output, N is the number of instances in the
                  image.
                - keypoint_scores (np.ndarray): the scores for all keypoints of
                  all instances with shape (N, K)
                - bbox_scores (np.ndarray): the predicted scores of the
                  bounding boxes with shape (N, )

            groundtruths (Sequence[dict]): A sequence of dict. If load from
                ``ann_file``, the dict inside can be empty. Else, each dict
                represents a groundtruths for an instance, with the following
                keys:

                - img_id (int): the image_id of the instance
                - width (int): the width of the image
                - height (int): the height of the image
                - crowd_index (float, optional): measures the crowding
                  level of an image defined in CrowdPose dataset
                - raw_ann_info (dict): the raw annotation information
                  of the instances, detailed below

                It is worth mentioning that, to compute
                    ``COCOKeyPointDetection``, there are some required keys in
                    the ``raw_ann_info``:

                - id: the id to distinguish different annotations
                - image_id: the image id of this annotation
                - category_id: the category of the instance
                - bbox: the bounding box of the instance
                - keypoints: the keypoints cooridinates with
                    their visibilities. It need to be aligned
                    with the official COCO format, e.g., a list
                    with length N * 3, in which N is the number of
                    keypoints, and each triplet represent the
                    [x, y, visible] of the keypoint.
                - iscrowd: indicating whether the annotation is
                    a crowd. It is useful when matching the
                    detection results to the ground truth.

                There are some optional keys as well:

                - area: it is necessary when ``self.use_area`` is
                    ``True``
                - num_keypoints: it is necessary when
                    ``self.iou_type`` is set as ``'keypoints_crowd'``
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'   # noqa: E501
            self._results.append((prediction, groundtruth))

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys for each ``gt_dict`` in ``gt_dicts``:

                - img_id: image id of the data sample
                - width: original image width
                - height: original image height
                - raw_ann_info: the raw annotation information,
                  detailed below

                Optional keys:

                - crowd_index: measure the crowding level of an image,
                  defined in CrowdPose dataset

                It is worth mentioning that, in order to compute
                ``CocoMetric``, there are some required keys in the
                ``raw_ann_info``:

                - id: the id to distinguish different annotations
                - image_id: the image id of this annotation
                - category_id: the category of the instance
                - bbox: the object bounding box
                - keypoints: the keypoints cooridinates along with their
                  visibilities. Note that it need to be aligned.
                  with the official COCO format, e.g., a list with length
                  N * 3, in which N is the number of keypoints, and each
                  triplet represent the [x, y, visible] of the keypoint.
                - iscrowd: indicating whether the annotation is a crowd.
                  It is useful when matching the detection results to
                  the ground truth.

                There are some optional keys as well:

                - area: it is necessary when ``self.use_area`` is ``True``
                - num_keypoints: it is necessary when ``self.iou_type``
                  is set as ``'keypoints_crowd'``

            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".

        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    assert 'area' in ann, \
                        '`area` is required when `self.use_area` is `True`'
                    annotation['area'] = ann['area']

                if self.iou_type == 'keypoints_crowd':
                    assert 'num_keypoints' in ann, \
                        '`num_keypoints` is required when `self.iou_type` ' \
                        'is `keypoints_crowd`'
                    annotation['num_keypoints'] = ann['num_keypoints']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmeval CocoMetric.')
        coco_json: dict = dict(
            info=info,
            images=image_infos,
            categories=self.dataset_meta['CLASSES'],
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        with open(converted_json_path, 'w') as f:
            dump(coco_json, f, sort_keys=True, indent=4)
        return converted_json_path

    def compute_metric(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        # split prediction and gt list
        preds, gts = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            # use converted gt json file to initialize coco helper
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)

        kpts = defaultdict(list)

        # group the preds by img_id
        for pred in preds:
            img_id = pred['img_id']
            for idx in range(len(pred['bbox_scores'])):
                instance = {
                    'id': pred['id'],
                    'img_id': pred['img_id'],
                    'category_id': pred.get('category_id', 1),
                    'keypoints': pred['keypoints'][idx],
                    'keypoint_scores': pred['keypoint_scores'][idx],
                    'bbox_score': pred['bbox_scores'][idx],
                }

                if 'areas' in pred:
                    instance['area'] = pred['areas'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    keypoints = pred['keypoints'][idx]
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts: dict = defaultdict(list)
        num_keypoints = self.dataset_meta['num_keypoints']
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                    instance['keypoints'], instance['keypoint_scores'][:, None]
                ],
                                                       axis=-1)
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score: float = 0.0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            elif self.nms_mode == 'oks_nms':
                keep = oks_nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]
            else:
                keep = soft_oks_nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        # evaluation results
        eval_results: OrderedDict = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list], outfile_prefix: str):
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json".
        """
        # the results with 'category_id'
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
            # collect all the person keypoints in current image
            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = [{
                'image_id': img_kpt['img_id'],
                'category_id': img_kpt['category_id'],
                'keypoints': keypoint.tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, keypoint in zip(img_kpts, _keypoints)]

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'
        with open(res_file, 'w') as f:
            dump(cat_results, f, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
                name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas,
                             self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == 'keypoints_crowd':
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)',
                'AP(M)', 'AP(H)'
            ]
        else:
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                'AR .75', 'AR (M)', 'AR (L)'
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self,
                                kpts: Dict[int, list],
                                key: str = 'id') -> Dict[int, list]:
        """Sort keypoint detection results in each image and remove the
        duplicate ones. Usually performed in multi-batch testing.

        Args:
            kpts (Dict[int, list]): keypoint prediction results. The keys are
                ``'img_id'`` and the values are list that may contain
                keypoints of multiple persons. Each element in the list is a
                dict containing the ``'key'`` field.
                See the argument ``key`` for details.
            key (str): The key name in each person prediction results. The
                corresponding value will be used for sorting the results.
                Defaults to ``'id'``.

        Returns:
            Dict[int, list]: The sorted keypoint detection results.
        """
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
