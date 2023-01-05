from PIL import Image
from PIL import ImageDraw
import numpy as np
import os

from coco_panoptic import COCOPanopticMetric

fake_dataset_metas = {
            'classes': ('person', 'dog', 'wall'),
            'thing_classes': ('person', 'dog'),
            'stuff_classes': ('wall', )
        }

if __name__ == '__main__':

    categories = fake_dataset_metas
    gt_img = Image.new("RGB",(256,256),(255,0,0))
    a = ImageDraw.ImageDraw(gt_img)  # draw the mask
    a.rectangle(((100, 100),(200, 200)), fill=(255,255,255))  

    white_id = 255 + 256 * 255 + 256 * 256 * 255
    red_id = 255
    gt_img_segm_info = [
        {"segments_info":
            [{
            'id': white_id,
            'category_id': 1,
            'iscrowd': 0,
            },
            {
            'id': red_id,
            'category_id': 0,
            'iscrowd': 0,
            }],
        "file_name": "gt.png"}
        ]
    if not os.path.exists('images'):
        os.makedirs('images')
    gt_img.save('./images/gt.png')
    gt_img_arr = np.array(gt_img, dtype=np.uint32)

    pred_img = Image.new("RGB",(256,256),(255,0,0))
    b = ImageDraw.ImageDraw(pred_img)  
    b.rectangle(((100, 100),(205, 195)), fill=(255,255,255)) 

    pred_img_arr = np.array(pred_img, dtype=np.uint32)

    pred_img_segm_info = [
        {"segments_info":
            [{
            'id': white_id,
            'category_id': 1,
            'iscrowd': 0,
            },
            {
            'id': red_id,
            'category_id': 0,
            'iscrowd': 0,
            }],
        "file_name": "pred.png"}
        ]

    pred_img.save('./images/pred.png')

    coco_pan_metric = COCOPanopticMetric(
        gt_folder='./images',
        pred_folder='./images',
        categories=categories,
        nproc=0  # use single thread
    )

    coco_pan_metric(predictions=pred_img_segm_info, groundtruths=gt_img_segm_info) 