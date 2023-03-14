# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import math
import numpy as np
import os.path as osp
import tempfile
from collections import OrderedDict
from json import dump
from multiprocessing import Process, Queue
from rich.console import Console
from rich.table import Table
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import get_text, load
from mmeval.metrics.utils import calculate_overlaps

PERSON_CLASSES = ['background', 'person']


class CrowdHuman(BaseMetric):
    """CrowdHuman evaluation metric.

    Evaluate Average Precision (AP), Miss Rate (MR) and Jaccard Index (JI) for
    detection tasks.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 eval_mode: int = 0,
                 iou_thrs: float = 0.5,
                 compare_matching_method: Optional[str] = None,
                 mr_ref: str = 'CALTECH_-2',
                 num_ji_process: int = 10,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.ann_file = ann_file

        # crowdhuman evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MR', 'AP', 'JI']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'MR', 'AP', 'JI', "
                               f'but got {metric}.')

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        self.outfile_prefix = outfile_prefix

        assert eval_mode in [0, 1, 2], \
            "Unknown eval mode. eval_mode should be one of '0', '1', '2'."

        assert compare_matching_method is None or \
               compare_matching_method == 'VOC', \
               'The alternative compare_matching_method is VOC.' \
               'This parameter defaults to CALTECH(None)'

        assert mr_ref in ['CALTECH_-2', 'CALTECH_-4'], \
            "mr_ref should be one of 'CALTECH_-2', 'CALTECH_-4'."

        self.eval_mode = eval_mode
        self.compare_matching_method = compare_matching_method
        self.mr_ref = mr_ref
        self.num_ji_process = num_ji_process
        self.backend_args = backend_args

        assert isinstance(iou_thrs, float), '`iou_thrs` should be float'
        self.iou_thrs = iou_thrs

    @staticmethod
    def results2json(results: Sequence[dict], outfile_prefix: str) -> str:
        """Dump the detection results to a COCO style json file.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json".

        Returns:
            str: The dump path of json file.
        """
        json_results_dict = dict()
        for pred, ann in results:
            dump_dict = dict()
            dump_dict['ID'] = ann['ID']
            dump_dict['width'] = ann['width']
            dump_dict['height'] = ann['height']

            bboxes = pred['bboxes'].tolist()
            scores = pred['scores']
            assert len(bboxes) == len(scores)
            dtboxes = []
            for i, single_bbox in enumerate(bboxes):
                temp_dict = dict()
                x1, y1, x2, y2 = single_bbox
                temp_dict['box'] = [x1, y1, x2 - x1, y2 - y1]
                temp_dict['score'] = float(scores[i])  # type: ignore
                temp_dict['tag'] = 1  # type: ignore
                dtboxes.append(temp_dict)
            dump_dict['dtboxes'] = dtboxes
            json_results_dict[ann['ID']] = dump_dict
        result_file = f'{outfile_prefix}.bbox.json'
        with open(result_file, 'w') as f:
            dump(json_results_dict, f)

        return result_file

    def add(self, predictions: Sequence[Dict], groundtruths: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - bboxes (np.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (np.ndarray): Shape (N, ), the predicted scores
                  of bounding boxes.

            groundtruths (Sequence[dict]): A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:

                - img_id (int): Image id.
                - width (int): The width of the image.
                - height (int): The height of the image.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                f'a sequence of dict, but got a sequence of {type(groundtruth)}.'   # noqa: E501
            self._results.append((prediction, groundtruth))

    def compute_metric(self, results: list) -> dict:
        """Compute the CrowdHuman metrics.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        # convert predictions to coco format and dump to json file
        result_file = self.results2json(results, outfile_prefix)

        eval_results: OrderedDict = OrderedDict()
        table_results: list = list()
        if self.format_only:
            self.logger.info(
                f'Results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        # load evaluation samples
        eval_samples = self.load_eval_samples(result_file)

        if 'AP' in self.metrics or 'MR' in self.metrics:
            score_list = self.compare(eval_samples)
            gt_num = sum([eval_samples[i].gt_num for i in eval_samples])
            ign_num = sum([eval_samples[i].ign_num for i in eval_samples])
            gt_num = gt_num - ign_num
            img_num = len(eval_samples)

        for metric in self.metrics:
            self.logger.info(f'Evaluating {metric}...')
            eval_func = getattr(self, f'eval_{metric.lower()}')

            if metric in ['AP', 'MR']:
                val = eval_func(score_list, gt_num, img_num)
            else:
                val = eval_func(eval_samples)

            eval_results[f'{metric}'] = float(val)
            table_results.append(f'{round(val * 100, 2):0.2f}')

        self._print_results(table_results)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def load_eval_samples(self, result_file: str) -> dict:
        """Load data from annotations file and detection results.

        Args:
            result_file (str): The file path of the saved detection results.

        Returns:
            dict: The detection result packaged by Image
        """
        gt_str = get_text(self.ann_file, backend_args=self.backend_args)
        gt_str_list = gt_str.strip().split('\n')
        gt_records = [json.loads(line) for line in gt_str_list]

        pred_records = load(result_file)
        eval_samples = dict()
        for gt_record in gt_records:
            img_id = gt_record['ID']
            assert img_id in pred_records, f'{img_id} is not in predictions'
            pred_record = pred_records[img_id]
            eval_samples[img_id] = Image(mode=self.eval_mode)
            eval_samples[img_id].load(
                record=gt_record,
                body_key='box',
                head_key=None,
                class_names=PERSON_CLASSES,
                gt_flag=True)
            eval_samples[img_id].load(
                record=pred_record,
                body_key='box',
                head_key=None,
                class_names=PERSON_CLASSES,
                gt_flag=False)
            eval_samples[img_id].clip_all_boader()
        return eval_samples

    def compare(self, samples: dict) -> list:
        """Match the detection results with the ground_truth.

        Args:
            samples (dict[Image]): The detection result packaged by Image.

        Returns:
            list: Matching result. A list of tuples (dtbox, label, imgID)
            in the descending sort of dtbox.score.
        """
        score_list = list()
        for id in samples.keys():
            if self.compare_matching_method == 'VOC':
                result = samples[id].compare_voc(self.iou_thrs)
            else:
                result = samples[id].compare_caltech(self.iou_thrs)
            score_list.extend(result)
        # In the descending sort of dtbox score.
        score_list.sort(key=lambda x: x[0][-1], reverse=True)
        return score_list

    @staticmethod
    def eval_ap(score_list: list, gt_num: int, img_num: int) -> float:
        """Compute Average Precision (AP).

        Args:
            score_list(list[tuple[ndarray, int, str]]): The matching result.
                a list of tuples (dtbox, label, imgID) in the descending
                sort of dtbox.score.
            gt_num(int): The number of gt boxes in the entire dataset.
            img_num(int): The number of images in the entire dataset.

        Returns:
            float: Average Precision (AP).
        """

        # calculate general ap score
        def _calculate_map(_recall: list, _precision: list) -> float:
            assert len(_recall) == len(_precision)
            area = 0
            for k in range(1, len(_recall)):
                delta_h = (_precision[k - 1] + _precision[k]) / 2
                delta_w = _recall[k] - _recall[k - 1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list()

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(score_list):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = gt_num - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / img_num)

        ap = _calculate_map(rpX, rpY)
        return ap

    def eval_mr(self, score_list: list, gt_num: int, img_num: int) -> float:
        """Compute Caltech-style log-average Miss Rate (MR).

        Args:
            score_list(list[tuple[ndarray, int, str]]): The matching result.
                a list of tuples (dtbox, label, imgID) in the descending
                sort of dtbox.score.
            gt_num(int): The number of gt boxes in the entire dataset.
            img_num(int): The number of images in the entire dataset.

        Returns:
            float: Miss Rate (MR).
        """

        # find greater_than
        def _find_gt(lst: list, target: float) -> int:
            for idx, _item in enumerate(lst):
                if _item >= target:
                    return idx
            return len(lst) - 1

        if self.mr_ref == 'CALTECH_-2':
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in
            # P.Dollar's paper
            ref = [
                0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162,
                0.5623, 1.000
            ]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in
            # S.Zhang's paper
            ref = [
                0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000,
                0.3162, 1.000
            ]

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list()
        for i, item in enumerate(score_list):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = gt_num - tp
            recall = tp / (tp + fn)
            missrate = 1.0 - recall
            fppi = fp / img_num
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        mr = np.exp(np.log(score).mean())
        return mr

    def eval_ji(self, samples: dict) -> float:
        """Compute Jaccard Index (JI) by using multi_process.

        Args:
            samples(Dict[str, Image]): The detection result packaged by Image.

        Returns:
            float: Jaccard Index (JI).
        """
        res_ji = []
        for i in range(10):
            score_thr = 1e-1 * i
            total = len(samples)
            stride = math.ceil(total / self.num_ji_process)
            records = list(samples.items())
            result_queue: Queue = Queue(10000)
            results, procs = [], []
            for j in range(self.num_ji_process):
                start = j * stride
                end = np.min([start + stride, total])
                sample_data = dict(records[start:end])
                p = Process(
                    target=self.compute_ji_with_ignore,
                    args=(result_queue, sample_data, score_thr))
                p.start()
                procs.append(p)
            for _ in range(total):
                t = result_queue.get()
                results.append(t)
            for p in procs:
                p.join()
            mean_ratio = self.gather(results)
            res_ji.append(mean_ratio)
        return max(res_ji)

    def compute_ji_with_ignore(self, result_queue: Queue, dt_result: dict,
                               score_thr: float):
        """Compute Jaccard Index (JI) with ignore.

        Args:
            result_queue (Queue): The Queue for save compute result when
                multi_process.
            dt_result (dict): Detection result packaged by Image.
            score_thr (float): The threshold of detection score.

        Returns:
            dict: compute result.
        """
        for ID, record in dt_result.items():
            gt_boxes = record.gt_boxes
            dt_boxes = record.dt_boxes
            keep = dt_boxes[:, -1] > score_thr
            dt_boxes = dt_boxes[keep][:, :-1]

            gt_tag = np.array(gt_boxes[:, -1] != -1)
            matches = self.compute_ji_matching(dt_boxes, gt_boxes[gt_tag, :4])
            # get the unmatched_indices
            matched_indices = np.array([j for (j, _) in matches])
            unmatched_indices = list(
                set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
            num_ignore_dt = self.get_ignores(dt_boxes[unmatched_indices],
                                             gt_boxes[~gt_tag, :4])
            matched_indices = np.array([j for (_, j) in matches])
            unmatched_indices = list(
                set(np.arange(gt_boxes[gt_tag].shape[0])) -
                set(matched_indices))
            num_ignore_gt = self.get_ignores(
                gt_boxes[gt_tag][unmatched_indices], gt_boxes[~gt_tag, :4])
            # compute results
            eps = 1e-6
            k = len(matches)
            m = gt_tag.sum() - num_ignore_gt
            n = dt_boxes.shape[0] - num_ignore_dt
            ratio = k / (m + n - k + eps)
            recall = k / (m + eps)
            cover = k / (n + eps)
            noise = 1 - cover
            result_dict = dict(
                ratio=ratio,
                recall=recall,
                cover=cover,
                noise=noise,
                k=k,
                m=m,
                n=n)
            result_queue.put_nowait(result_dict)

    @staticmethod
    def gather(results: list) -> float:
        """Integrate test results.

        Args:
            results (list): A list of compute results.

        Returns:
            float: Compute result.
        """
        assert len(results)
        img_num = 0
        for result in results:
            if result['n'] != 0 or result['m'] != 0:
                img_num += 1
        mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
        return mean_ratio

    def compute_ji_matching(self, dt_boxes: np.ndarray,
                            gt_boxes: np.ndarray) -> list:
        """Match the annotation box for each detection box.

        Args:
            dt_boxes(ndarray): Detection boxes.
            gt_boxes(ndarray): Ground_truth boxes.

        Returns:
            list: Match result.
        """
        assert dt_boxes.shape[-1] > 3 and gt_boxes.shape[-1] > 3
        if dt_boxes.shape[0] < 1 or gt_boxes.shape[0] < 1:
            return list()

        ious = calculate_overlaps(dt_boxes, gt_boxes, mode='iou')
        input_ = copy.deepcopy(ious)
        input_[input_ < self.iou_thrs] = 0
        match_scipy = maximum_bipartite_matching(
            csr_matrix(input_), perm_type='column')
        matches = []
        for i in range(len(match_scipy)):
            if match_scipy[i] != -1:
                matches.append((i, int(match_scipy[i])))
        return matches

    def get_ignores(self, dt_boxes: np.ndarray, gt_boxes: np.ndarray) -> int:
        """Get the number of ignore bboxes.

        Args:
            dt_boxes(ndarray): Detection boxes.
            gt_boxes(ndarray): Ground_truth boxes.

        Returns:
            int: Number of ignored boxes.
        """
        if gt_boxes.size:
            ioas = calculate_overlaps(dt_boxes, gt_boxes, mode='iof')
            ioas = np.max(ioas, axis=1)
            rows = np.where(ioas > self.iou_thrs)[0]
            return len(rows)
        else:
            return 0

    def _print_results(self, table_results: list) -> None:
        """Print the evaluation results table.

        Args:
            table_results (list): The computed metric.
        """
        table_title = 'CrowdHuman Results IoU=' \
                      f'{round(self.iou_thrs * 100)} (%)'

        table = Table(title=table_title, width=50)
        console = Console(width=100)
        for name in self.metrics:
            table.add_column(name, justify='left')
        table.add_row(*table_results)
        with console.capture() as capture:
            console.print(table, end='')
        self.logger.info('\n' + capture.get())


class Image:
    """Data structure for evaluation of CrowdHuman.

    Note:
        This implementation is modified from
        https://github.com/Purkialo/CrowdDet/blob/master/lib/evaluate/APMRToolkits/image.py   # noqa: E501

    Args:
        mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
    """

    def __init__(self, mode: int) -> None:
        self.ID = None
        self.width = None
        self.height = None
        self.dt_boxes = None
        self.gt_boxes = None
        self.eval_mode = mode

        self.ign_num = None
        self.gt_num = None
        self.dt_num = None

    def load(self, record: dict, body_key: Optional[str],
             head_key: Optional[str], class_names: list,
             gt_flag: bool) -> None:
        """Loading information for evaluation.

        Args:
            record (dict): Label information or prediction results.
                The format might look something like this:
                {
                    'ID': '273271,c9db000d5146c15',
                    'gtboxes': [
                        {'fbox': [72, 202, 163, 503], 'tag': 'person', ...},
                        {'fbox': [199, 180, 144, 499], 'tag': 'person', ...},
                        ...
                    ]
                }
                or:
                {
                    'ID': '273271,c9db000d5146c15',
                    'width': 800,
                    'height': 1067,
                    'dtboxes': [
                        {
                            'box': [306.22, 205.95, 164.05, 394.04],
                            'score': 0.99,
                            'tag': 1
                        },
                        {
                            'box': [403.60, 178.66, 157.15, 421.33],
                            'score': 0.99,
                            'tag': 1
                        },
                        ...
                    ]
                }
            body_key (str, None): key of detection body box.
                Valid when loading detection results and self.eval_mode!=1.
            head_key (str, None): key of detection head box.
                Valid when loading detection results and self.eval_mode!=0.
            class_names (list[str]):class names of data set.
                Defaults to ['background', 'person'].
            gt_flag (bool): Indicate whether record is ground truth
                or predicting the outcome.
        """
        if 'ID' in record and self.ID is None:
            self.ID = record['ID']
        if 'width' in record and self.width is None:
            self.width = record['width']
        if 'height' in record and self.height is None:
            self.height = record['height']
        if gt_flag:
            self.gt_num = len(record['gtboxes'])  # type: ignore
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes',
                                                      class_names)
            if self.eval_mode == 0:
                self.gt_boxes = body_bbox
                self.ign_num = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gt_boxes = head_bbox
                self.ign_num = (head_bbox[:, -1] == -1).sum()
            else:
                gt_tag = np.array([
                    body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                    for i in range(len(body_bbox))
                ])
                self.ign_num = (gt_tag == 0).sum()
                self.gt_boxes = np.hstack(
                    (body_bbox[:, :-1], head_bbox[:, :-1],
                     gt_tag.reshape(-1, 1)))

        if not gt_flag:
            self.dt_num = len(record['dtboxes'])  # type: ignore
            if self.eval_mode == 0:
                self.dt_boxes = self.load_det_boxes(record, 'dtboxes',
                                                    body_key, 'score')
            elif self.eval_mode == 1:
                self.dt_boxes = self.load_det_boxes(record, 'dtboxes',
                                                    head_key, 'score')
            else:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key,
                                                   'score')
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key,
                                                   'score')
                self.dt_boxes = np.hstack((body_dtboxes, head_dtboxes))

    @staticmethod
    def load_gt_boxes(dict_input, key_name, class_names):
        """load ground_truth and transform [x, y, w, h] to [x1, y1, x2, y2]"""
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = copy.deepcopy(body_tag)
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            body_bbox.append(np.hstack((rb['fbox'], body_tag)))
        head_bbox = np.array(head_bbox)
        head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    @staticmethod
    def load_det_boxes(dict_input, key_name, key_box, key_score, key_tag=None):
        """load detection boxes."""
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_score], rb[key_tag]))
                    for rb in dict_input[key_name]
                ])
            else:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_score]))
                    for rb in dict_input[key_name]
                ])
        else:
            if key_tag:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_tag]))
                    for rb in dict_input[key_name]
                ])
            else:
                bboxes = np.vstack(
                    [rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def clip_all_boader(self) -> None:
        """Make sure boxes are within the image range."""

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dt_boxes.shape[-1] >= 4  # type: ignore
        assert self.gt_boxes.shape[-1] >= 4  # type: ignore
        assert self.width is not None and self.height is not None
        if self.eval_mode == 2:
            self.dt_boxes[:, :4] = _clip_boundary(self.dt_boxes[:, :4],
                                                  self.height, self.width)
            self.gt_boxes[:, :4] = _clip_boundary(self.gt_boxes[:, :4],
                                                  self.height, self.width)
            self.dt_boxes[:, 4:8] = _clip_boundary(self.dt_boxes[:, 4:8],
                                                   self.height, self.width)
            self.gt_boxes[:, 4:8] = _clip_boundary(self.gt_boxes[:, 4:8],
                                                   self.height, self.width)
        else:
            self.dt_boxes = _clip_boundary(self.dt_boxes, self.height,
                                           self.width)
            self.gt_boxes = _clip_boundary(self.gt_boxes, self.height,
                                           self.width)

    def compare_voc(self,
                    iou_thrs: float) -> List[Tuple[np.ndarray, int, str]]:
        """Match the detection results with the ground_truth by VOC.

        Args:
            iou_thrs (float): IOU threshold.

        Returns:
            list[tuple[np.ndarray, int, str]]: Matching result.
            A list of tuple (dtbox, label, imgID) in the descending
            sort of dtbox.score.
        """
        if self.dt_boxes is None:
            return list()
        dtboxes = self.dt_boxes
        gtboxes = self.gt_boxes if self.gt_boxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        score_list = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = iou_thrs

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    score_list.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                score_list.append((dt, self.ID))
        return score_list

    def compare_caltech(self,
                        iou_thrs: float) -> List[Tuple[np.ndarray, int, str]]:
        """Match the detection results with the ground_truth by Caltech
        matching strategy.

        Args:
            iou_thrs (float): IOU threshold.

        Returns:
            list[tuple[np.ndarray, int, str]]: Matching result.
            A list of tuple (dtbox, label, imgID) in the descending
            sort of dtbox.score.
        """
        if self.dt_boxes is None or self.gt_boxes is None:
            return list()

        dtboxes = self.dt_boxes if self.dt_boxes is not None else list()
        gtboxes = self.gt_boxes if self.gt_boxes is not None else list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = calculate_overlaps(dtboxes, gtboxes, mode='iou')
            overlap_ioa = calculate_overlaps(dtboxes, gtboxes, mode='iof')
        else:
            return list()

        score_list = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = iou_thrs
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > iou_thrs:
                            maxiou = overlap
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    score_list.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                score_list.append((dt, 0, self.ID))
        return score_list
