# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np


def calculate_average_precision(recalls: np.ndarray,
                                precisions: np.ndarray,
                                mode: str = 'area') -> float:
    """Calculate average precision in the detection task.

    Args:
        recalls (ndarray): The recalls with shape (num_dets, ).
        precisions (ndarray): The precisions with shape (num_dets, ).
        mode (str): The average mode, should be 'area' or '11points'.
            'area' means calculating the area under precision-recall curve.
            '11points' means calculating the average precision of recalls at
            [0, 0.1, ..., 1.0]. Defaults to 'area'.

    Returns:
        float: Calculated average precision.
    """
    assert mode in ['area', '11points']
    if mode == 'area':
        mrec = np.hstack((0, recalls, 1))
        mpre = np.hstack((0, precisions, 0))
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        ind = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[ind + 1] - mrec[ind]) * mpre[ind + 1])
    else:
        ap = 0.0
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[recalls >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11.0
    return ap
