'''
Author: Ghost 1432072586@qq.com
Date: 2022-12-01 16:30:30
LastEditors: Ghost 1432072586@qq.com
LastEditTime: 2022-12-01 16:39:54
FilePath: /mmeval/mmeval/metrics/utils/bbox_iou_rotated.py
Description: 
'''
import numpy as np


def bbox_iou_rotated(bboxes1: np.ndarray,
                     bboxes2: np.ndarray,
                     mode: str = 'iou',
                     aligned: bool = False,
                     clockwise: bool = True) -> np.ndarray:
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = np.zeros(rows)
    else:
        ious = np.zeros(rows * cols)
    if not clockwise:
        flip_mat = np.ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat

    if not aligned:
        ious = ious.view(rows, cols)
    return ious