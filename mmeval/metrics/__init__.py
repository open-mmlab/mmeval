# Copyright (c) OpenMMLab. All rights reserved.

import warnings

from .accuracy import Accuracy
from .ava_map import AVAMeanAP
from .average_precision import AveragePrecision
from .bleu import BLEU
from .char_recall_precision import CharRecallPrecision
from .coco_detection import COCODetection
from .connectivity_error import ConnectivityError
from .dota_map import DOTAMeanAP
from .end_point_error import EndPointError
from .f1_score import F1Score
from .gradient_error import GradientError
from .hmean_iou import HmeanIoU
from .keypoint_auc import KeypointAUC
from .keypoint_epe import KeypointEndPointError
from .keypoint_nme import KeypointNME
from .mae import MeanAbsoluteError
from .matting_mse import MattingMeanSquaredError
from .mean_iou import MeanIoU
from .mse import MeanSquaredError
from .niqe import NaturalImageQualityEvaluator
from .oid_map import OIDMeanAP
from .pck_accuracy import JhmdbPCKAccuracy, MpiiPCKAccuracy, PCKAccuracy
from .perplexity import Perplexity
from .precision_recall_f1score import (MultiLabelPrecisionRecallF1score,
                                       PrecisionRecallF1score,
                                       SingleLabelPrecisionRecallF1score)
from .proposal_recall import ProposalRecall
from .psnr import PeakSignalNoiseRatio
from .rouge import ROUGE
from .sad import SumAbsoluteDifferences
from .snr import SignalNoiseRatio
from .ssim import StructuralSimilarity
from .voc_map import VOCMeanAP
from .word_accuracy import WordAccuracy

__all__ = [
    'Accuracy', 'MeanIoU', 'VOCMeanAP', 'OIDMeanAP', 'EndPointError',
    'F1Score', 'HmeanIoU', 'COCODetection', 'PCKAccuracy', 'MpiiPCKAccuracy',
    'JhmdbPCKAccuracy', 'ProposalRecall', 'PeakSignalNoiseRatio',
    'MeanAbsoluteError', 'MeanSquaredError', 'StructuralSimilarity',
    'SignalNoiseRatio', 'AveragePrecision', 'AVAMeanAP', 'BLEU', 'DOTAMeanAP',
    'SumAbsoluteDifferences', 'GradientError', 'MattingMeanSquaredError',
    'ConnectivityError', 'ROUGE', 'Perplexity', 'KeypointEndPointError',
    'KeypointAUC', 'KeypointNME', 'NaturalImageQualityEvaluator',
    'WordAccuracy', 'PrecisionRecallF1score',
    'SingleLabelPrecisionRecallF1score', 'MultiLabelPrecisionRecallF1score',
    'CharRecallPrecision'
]

_deprecated_msg = (
    '`{n1}` is a deprecated metric alias for `{n2}`. '
    'To silence this warning, use `{n2}` by itself. '
    'The deprecated metric alias would be removed in mmeval 1.0.0!')

__deprecated_metric_names__ = {
    'COCODetectionMetric': 'COCODetection',
    'F1Metric': 'F1Score',
    'MAE': 'MeanAbsoluteError',
    'MSE': 'MeanSquaredError',
    'PSNR': 'PeakSignalNoiseRatio',
    'SNR': 'SignalNoiseRatio',
    'SSIM': 'StructuralSimilarity',
    'SAD': 'SumAbsoluteDifferences',
    'MattingMSE': 'MattingMeanSquaredError',
    'SingleLabelMetric': 'SingleLabelPrecisionRecallF1score',
    'MultiLabelMetric': 'MultiLabelPrecisionRecallF1score'
}


def __getattr__(attr: str):
    """Customization of module attribute access.

    Thanks to pep-0562, we can customize moudel's attribute access
    via ``__getattr__`` to implement deprecation warnings.

    With this function, we can implement the following features:

        >>> from mmeval.metrics import COCODetectionMetric
        <stdin>:1: DeprecationWarning: `COCODetectionMetric` is a deprecated
        metric alias for `COCODetection`. To silence this warning, use
        `COCODetection` by itself. The deprecated metric alias would be
        removed in mmeval 1.0.0!
    """
    if attr in __deprecated_metric_names__:
        message = _deprecated_msg.format(
            n1=attr, n2=__deprecated_metric_names__[attr])
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return globals()[__deprecated_metric_names__[attr]]
    raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')
