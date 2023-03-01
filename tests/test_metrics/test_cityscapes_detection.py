import math
import numpy as np
import os
import os.path as osp
import pytest
import tempfile
from PIL import Image

from mmeval.metrics import CityScapesDetection
from mmeval.utils import try_import

cityscapesscripts = try_import('cityscapesscripts')


def _gen_fake_datasamples(seg_prefix):
    city = 'lindau'
    os.makedirs(osp.join(seg_prefix, city), exist_ok=True)

    sequenceNb = '000000'
    frameNb1 = '000019'
    img_name1 = f'{city}_{sequenceNb}_{frameNb1}_gtFine_instanceIds.png'
    img_path1 = osp.join(seg_prefix, city, img_name1)
    basename1 = osp.splitext(osp.basename(img_path1))[0]
    masks1 = np.zeros((20, 20), dtype=np.int32)
    masks1[:10, :10] = 24 * 1000
    Image.fromarray(masks1).save(img_path1)

    dummy_mask1 = np.zeros((1, 20, 20), dtype=np.uint8)
    dummy_mask1[:, :10, :10] = 1
    prediction1 = {
        'basename': basename1,
        'mask_scores': np.array([1.0]),
        'labels': np.array([0]),
        'masks': dummy_mask1
    }
    groundtruth1 = {'file_name': img_path1}

    frameNb2 = '000020'
    img_name2 = f'{city}_{sequenceNb}_{frameNb2}_gtFine_instanceIds.png'
    img_path2 = osp.join(seg_prefix, city, img_name2)
    basename2 = osp.splitext(osp.basename(img_path2))[0]
    masks2 = np.zeros((20, 20), dtype=np.int32)
    masks2[:10, :10] = 24 * 1000 + 1
    Image.fromarray(masks2).save(img_path2)

    dummy_mask2 = np.zeros((1, 20, 20), dtype=np.uint8)
    dummy_mask2[:, :10, :10] = 1
    prediction2 = {
        'basename': basename2,
        'mask_scores': np.array([0.98]),
        'labels': np.array([1]),
        'masks': dummy_mask1
    }
    groundtruth2 = {'file_name': img_path2}
    return [prediction1, prediction2], [groundtruth1, groundtruth2]


@pytest.mark.skipif(
    cityscapesscripts is None, reason='cityscapesscriptsr is not available!')
def test_metric_invalid_usage():
    with pytest.raises(AssertionError):
        CityScapesDetection(outfile_prefix=None)

    with pytest.raises(AssertionError):
        CityScapesDetection(
            outfile_prefix='tmp/cityscapes/results', seg_prefix=None)

    with pytest.raises(AssertionError):
        CityScapesDetection(
            outfile_prefix='tmp/cityscapes/results', seg_prefix=None)

    with pytest.raises(AssertionError):
        CityScapesDetection(
            outfile_prefix='tmp/cityscapes/results',
            format_only=True,
            keep_results=False)


@pytest.mark.skipif(
    cityscapesscripts is None, reason='cityscapesscriptsr is not available!')
def test_compute_metric():
    tmp_dir = tempfile.TemporaryDirectory()

    dataset_metas = {
        'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle')
    }

    # create dummy data
    seg_prefix = osp.join(tmp_dir.name, 'cityscapes', 'gtFine', 'val')
    os.makedirs(seg_prefix, exist_ok=True)
    predictions, groundtruths = _gen_fake_datasamples(seg_prefix)

    # test single cityscapes dataset evaluation
    cityscapes_det_metric = CityScapesDetection(
        dataset_meta=dataset_metas,
        outfile_prefix=osp.join(tmp_dir.name, 'test'),
        seg_prefix=seg_prefix,
        keep_results=False,
        keep_gt_json=False,
        classwise=False)

    eval_results = cityscapes_det_metric(
        predictions=predictions, groundtruths=groundtruths)
    target = {'mAP': 0.5, 'AP50': 0.5}
    eval_results.pop('results_list')
    assert eval_results == target

    # test classwise result evaluation
    cityscapes_det_metric = CityScapesDetection(
        dataset_meta=dataset_metas,
        outfile_prefix=osp.join(tmp_dir.name, 'test'),
        seg_prefix=seg_prefix,
        keep_results=False,
        keep_gt_json=False,
        classwise=True)

    eval_results = cityscapes_det_metric(
        predictions=predictions, groundtruths=groundtruths)
    eval_results.pop('results_list')
    mAP = eval_results.pop('mAP')
    AP50 = eval_results.pop('AP50')
    person_ap = eval_results.pop('person_ap')
    person_ap50 = eval_results.pop('person_ap50')
    assert mAP == 0.5
    assert AP50 == 0.5
    assert person_ap == 0.5
    assert person_ap50 == 0.5
    # others classes ap or ap50 should be nan
    for v in eval_results.values():
        assert math.isnan(v)

    # test format only evaluation
    cityscapes_det_metric = CityScapesDetection(
        dataset_meta=dataset_metas,
        format_only=True,
        outfile_prefix=osp.join(tmp_dir.name, 'test'),
        seg_prefix=seg_prefix,
        keep_results=True,
        keep_gt_json=True,
        classwise=True)

    eval_results = cityscapes_det_metric(
        predictions=predictions, groundtruths=groundtruths)
    assert osp.exists(f'{osp.join(tmp_dir.name, "test")}')
    assert eval_results == dict()

    tmp_dir.cleanup()
