# Run detection on a COCO-format dataset
# Only at a fixed rate, not on every frame
raise Exception('Not updated with new detection interface')
import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob

from tqdm import tqdm
import numpy as np

import torch
import mmcv

from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from dbcode.dbinfo import coco2av, coco2kmots, kmots_classes, av_classes
from det import parse_mmdet_result, vis_det, eval_ccf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--det-stride', type=int, default=30)
    parser.add_argument('--in-scale', type=float, required=True)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)

    # parser.add_argument('--config', type=str, default='../mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    # parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    
    parser.add_argument('--no-eval', action='store_true', default=False)
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

    db = COCO(opts.annot_path)
    n_class = len(db.cats)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']
    if 'KMOTS' in opts.data_root:
        class_mapping = coco2kmots
        class_names = kmots_classes
    elif 'ArgoVerse' in opts.data_root:
        class_mapping = coco2av
        class_names = av_classes
    else:
        raise Exception('Unknown dataset')

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

    results_ccf = []

    for iid, img in tqdm(db.imgs.items()):
        img_name = img['name']

        sid = img['sid']
        seq_name = seqs[sid]

        img_path = join(opts.data_root, seq_dirs[sid], img_name)
        I = mmcv.imread(img_path)
        
        if iid % opts.det_stride == 0:
            result = inference_detector(model, I)
            bboxes, scores, labels, masks = parse_mmdet_result(result, class_mapping, n_class)
            if len(bboxes):
                bboxes_ltwh = bboxes.copy()
                        # convert to coco fmt
                bboxes_ltwh[:, 2:] -= bboxes_ltwh[:, :2]
                bboxes_ltwh = bboxes_ltwh.tolist()
            else:
                bboxes_ltwh = []

        if vis_out:
            vis_path = join(opts.vis_dir, seq_name, img_name[:-3] + 'jpg')
            if opts.overwrite or not isfile(vis_path):
                vis_det(
                    I, bboxes, labels,
                    class_names, masks, scores,
                    out_scale=opts.vis_scale,
                    out_file=vis_path
                )

        for i in range(len(bboxes_ltwh)):
            result_dict = {
                'image_id': iid,
                'bbox': bboxes_ltwh[i],
                'score': scores[i],
                'category_id': labels[i],
            }
            if masks is not None:
                result_dict['segmentation'] = masks[i]
            results_ccf.append(result_dict)

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}"')

if __name__ == '__main__':
    main()