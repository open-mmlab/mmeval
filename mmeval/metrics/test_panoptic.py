from PIL import Image
from PIL import ImageDraw
import numpy as np

from coco_panoptic import COCOPanopticMetric

fake_dataset_metas = [
    {
        "name": '0',
        "isthing": 0
    },
    {
        "name": '1',
        "isthing": 1
    }
]

if __name__ == '__main__':

    categories = fake_dataset_metas

    coco_pan_metric = COCOPanopticMetric(
        categories=categories,
        nproc=0
    )

    gt_img = Image.new("RGB",(256,256),(255,0,0))
    a = ImageDraw.ImageDraw(gt_img)  # draw the mask
    a.rectangle(((100, 100),(200, 200)), fill=(255,255,255))  

    white_id = 255 + 256 * 255 + 256 * 256 * 255
    red_id = 255
    gt_img_segm_info = [
        {
        'id': white_id,
        'category_id': 1,
        'iscrowd': 0,
        },
        {
        'id': red_id,
        'category_id': 0,
        'iscrowd': 0,
        }]

    gt_img_arr = np.array(gt_img, dtype=np.uint32)


    pred_img = Image.new("RGB",(256,256),(255,0,0))
    b = ImageDraw.ImageDraw(pred_img)  
    b.rectangle(((100, 100),(205, 195)), fill=(255,255,255)) 


    pred_img_arr = np.array(pred_img, dtype=np.uint32)

    pred_img_segm_info = [
        {
        'id': white_id,
        'category_id': 1,
        'iscrowd': 0,
        },
        {
        'id': red_id,
        'category_id': 0,
        'iscrowd': 0,
        }]

    groundtruth_img_list = [gt_img_arr,]
    prediction_img_list = [pred_img_arr,]

    coco_pan_metric(image_pairs=(groundtruth_img_list, prediction_img_list), ann_pairs=([gt_img_segm_info,], [pred_img_segm_info,])) 