# Run detection on every frame for clips with annotation in MEVA dataset

import argparse, pickle
from os.path import join, isfile, basename
from glob import glob

from tqdm import tqdm
import numpy as np

import torch
import mmcv

from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector

import pycocotools.mask as maskUtils

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
# from det import vis_mmdet

def parse_args():
    parser = argparse.ArgumentParser(description='Run detector on a folder of images')
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--image-stride', type=int, default=1)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--vis-class-subset', nargs='+', type=int, default=[0, 2, 5, 7])
    parser.add_argument('--config', type=str, default='../../repo/mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    config = mmcv.Config.fromfile(opts.config)
    if opts.in_scale is not None:
        config.data.test.img_scale = opts.in_scale
        # mainly for SSD
        config.data.test.resize_keep_ratio = True
    if opts.no_mask:
        if 'mask_head' in config.model:
            config.model['mask_head'] = None

    model = init_detector(config, opts.weights)
    model.eval()

    img_list = sorted(glob(join(opts.in_dir, '*.jpg')))
    img_list = img_list[::opts.image_stride]

    for img_path in tqdm(img_list):
        img_name = basename(img_path)
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)
        out_path = join(opts.out_dir, img_name[:-3] + 'pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(result, open(out_path, 'wb'))
        # if vis_out:
        #     vis_path = join(opts.vis_dir, img_name[:-3] + 'jpg')
        #     if opts.overwrite or not isfile(vis_path):
        #         vis_mmdet(img, result, coco_classes, 
        #             opts.vis_class_subset, opts.vis_scale
        #             out_file=vis_path
        #         )

if __name__ == '__main__':
    main()