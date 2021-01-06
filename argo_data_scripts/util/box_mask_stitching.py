''' 
Real-time location + offline shape
'''

import argparse, json, pickle
from os.path import join, isfile
from time import perf_counter

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from util.bbox import ltwh2ltrb
from det import imread, vis_det, eval_ccf
from track import iou_assoc
from forecast import warp_mask_to_box


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=int, default=0, help='eta >= -1')
    parser.add_argument('--assoc', type=str, default='iou')
    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--box-ccf-path', type=str, required=True)
    parser.add_argument('--mask-ccf-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    box_ccf = pickle.load(open(opts.box_ccf_path, 'rb'))
    mask_ccf = pickle.load(open(opts.mask_ccf_path, 'rb'))
    box_end_idx = 0
    mask_end_idx = 0

    results_ccf = []        # instance based

    for iid, img in tqdm(db.imgs.items()):
        img_name = img['name']

        sid = img['sid']
        seq_name = seqs[sid]
        
        box_start_idx = box_end_idx
        while box_start_idx < len(box_ccf) and box_ccf[box_start_idx]['image_id'] < img['id']:
            box_start_idx += 1
        box_end_idx = box_start_idx
        while box_end_idx < len(box_ccf) and box_ccf[box_end_idx]['image_id'] == img['id']:
            box_end_idx += 1
        box_dets = box_ccf[box_start_idx:box_end_idx]
        
        mask_start_idx = mask_end_idx
        while mask_start_idx < len(mask_ccf) and mask_ccf[mask_start_idx]['image_id'] < img['id']:
            mask_start_idx += 1
        mask_end_idx = mask_start_idx
        while mask_end_idx < len(mask_ccf) and mask_ccf[mask_end_idx]['image_id'] == img['id']:
            mask_end_idx += 1
        mask_dets = mask_ccf[mask_start_idx:mask_end_idx]

        if len(box_dets) == 0:
            bboxes1, scores1, labels1, masks1 = [], [], [], None
        elif len(mask_dets) == 0:
            bboxes1 = np.array([d['bbox'] for d in box_dets])
            scores1 = np.array([d['score'] for d in box_dets])
            labels1 = np.array([d['category_id'] for d in box_dets])
            masks1 = np.array([d['segmentation'] for d in box_dets])
        else:
            # the slow version, but works with out of order ccf results
            # dets = [r for r in box_ccf if r['image_id'] == img['id']]
            
            bboxes1 = np.array([d['bbox'] for d in box_dets])
            scores1 = np.array([d['score'] for d in box_dets])
            labels1 = np.array([d['category_id'] for d in box_dets])
            masks1 = np.array([d['segmentation'] for d in box_dets])
            
            bboxes2 = np.array([d['bbox'] for d in mask_dets])
            scores2 = np.array([d['score'] for d in mask_dets])
            labels2 = np.array([d['category_id'] for d in mask_dets])
            masks2 = np.array([d['segmentation'] for d in mask_dets])

            score_argsort = np.argsort(scores1)[::-1]
            bboxes1 = bboxes1[score_argsort]
            scores1 = scores1[score_argsort]
            labels1 = labels1[score_argsort]
            masks1 = masks1[score_argsort]
        
            score_argsort = np.argsort(scores2)[::-1]
            bboxes2 = bboxes2[score_argsort]
            scores2 = scores2[score_argsort]
            labels2 = labels2[score_argsort]
            masks2 = masks2[score_argsort]

            order1, order2, n_matched12, _, _ = iou_assoc(
                bboxes1, labels1, np.arange(len(bboxes1)), 0,
                bboxes2, labels2, opts.match_iou_th,
                no_unmatched1=False,
            )

            bboxes1 = bboxes1[order1]
            scores1 = scores1[order1]
            labels1 = labels1[order1]
            masks1 = masks1[order1]

            bboxes2 = bboxes2[order2]
            scores2 = scores2[order2]
            labels2 = labels2[order2]
            masks2 = masks2[order2]

            mask1_fix = warp_mask_to_box(
                masks2[:n_matched12],
                bboxes2[:n_matched12],
                bboxes1[:n_matched12]
            )

            masks1 = np.concatenate((mask1_fix, masks1[n_matched12:]))

        if vis_out:
            img_path = join(opts.data_root, seq_dirs[sid], img_name)
            I = imread(img_path)
            vis_path = join(opts.vis_dir, seq_name, img_name[:-3] + 'jpg')
            bboxes = ltwh2ltrb(bboxes1) if len(bboxes1) else []
            if opts.overwrite or not isfile(vis_path):
                vis_det(
                    I, bboxes, labels1,
                    class_names, masks1, scores1,
                    out_scale=opts.vis_scale,
                    out_file=vis_path
                )

        n = len(bboxes1)
        for i in range(n):
            result_dict = {
                'image_id': iid,
                'bbox': bboxes1[i],
                'score': scores1[i],
                'category_id': labels1[i],
            }
            if masks1 is not None:
                result_dict['segmentation'] = masks1[i]
            results_ccf.append(result_dict)


    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}"')

if __name__ == '__main__':
    main()