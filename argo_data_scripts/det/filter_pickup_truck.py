# Filtering-out simultaneous car and truck detection (confusion cause by pickup trucks)


import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from det import imread, parse_det_result, eval_ccf, vis_det



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    # parser.add_argument('--data-root', type=str, default='/data2/mengtial/ArgoVerse1.1/tracking')
    # parser.add_argument('--annot-path', type=str, default='/data2/mengtial/ArgoVerse1.1/tracking/coco_fmt/val_c3.json')
    # parser.add_argument('--vis-dir', type=str, default='/data2/mengtial/Exp/ArgoVerse1.1-debug/pps_mrcnn50_nm_ds_s0.75_fba_iou_lin_pkt')
    # parser.add_argument('--vis-scale', type=float, default=1)
    # parser.add_argument('--in-dir', type=str, default='/data2/mengtial/Exp/ArgoVerse1.1-c3-eta0/output/pps_mrcnn50_nm_ds_s0.75_fba_iou_lin/val')
    # parser.add_argument('--out-dir', type=str, default='/data2/mengtial/Exp/ArgoVerse1.1-c3-eta0/output/pps_mrcnn50_nm_ds_s0.75_fba_iou_lin_pkt/val')
    # parser.add_argument('--no-eval', action='store_true', default=True)
    # parser.add_argument('--overwrite', action='store_true', default=True)


    opts = parser.parse_args()
    return opts


def remove_car_for_pickup(bboxes, labels, scores, match_iou_th=0.7, score_th=0.3):
    # bboxes are in the form of a list of [l, t, w, h]
    # assuming c3
    truck_idx = np.where(np.logical_and(labels == 5, scores >= score_th))[0]
    n_truck = len(truck_idx)
    if n_truck == 0:
        return []
    car_idx = np.where(labels == 2)[0]
    n_car = len(car_idx)
    if n_car == 0:
        return [] 

    trucks = bboxes[truck_idx]
    cars = bboxes[car_idx]

    _ = n_car*[0]
    ious = maskUtils.iou(trucks, cars, _)

    car_matched = n_car*[None]
    remove = []

    for i in range(n_truck):
        best_iou = match_iou_th
        match_j = None
        for j in range(n_car):
            if car_matched[j] is not None \
                or ious[i, j] < best_iou:
                continue
            best_iou = ious[i, j]
            match_j = j
        if match_j is not None:
            remove.append(car_idx[match_j])

    return remove

def main():
    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    assert class_names[5] == 'truck'
    assert class_names[2] == 'car'
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    in_ccf = pickle.load(open(join(opts.in_dir, 'results_ccf.pkl'), 'rb'))
    results_ccf = []
  
    i_start = 0
    for sid, seq in enumerate(tqdm(seqs)):
        # if seq != '5ab2697b-6e3e-3454-a36a-aba2c6f27818':
        #     continue
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        for ii, img in enumerate(frame_list):
            iid = img['id']
            i_end = i_start
            while i_end < len(in_ccf) and in_ccf[i_end]['image_id'] == iid:
                i_end += 1
            dets = in_ccf[i_start:i_end]
            i_start = i_end
            # dets = [r for r in in_ccf if r['image_id'] == img['id']]
            bboxes = np.array([d['bbox'] for d in dets])
            labels = np.array([d['category_id'] for d in dets])
            scores = np.array([d['score'] for d in dets])
            with_mask = len(dets) and 'segmentation' in dets[0]
            if with_mask:
                masks = np.array([d['segmentation'] for d in dets])

            score_argsort = np.argsort(scores)[::-1]
            bboxes = bboxes[score_argsort]
            scores = scores[score_argsort]
            labels = labels[score_argsort]
            if with_mask:
                masks = masks[score_argsort]

            remove_idx = remove_car_for_pickup(bboxes, labels, scores)
            bboxes = np.delete(bboxes, remove_idx, 0)
            labels = np.delete(labels, remove_idx, 0)
            scores = np.delete(scores, remove_idx, 0)
            if with_mask:
                masks = np.delete(masks, remove_idx, 0)
            n = len(bboxes)
            for i in range(n):
                result_dict = {
                    'image_id': img['id'],
                    'bbox': bboxes[i],
                    'score': scores[i],
                    'category_id': labels[i],
                }
                if with_mask:
                    result_dict['segmentation'] = masks[i]
                results_ccf.append(result_dict)

            if vis_out:
                img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                I = imread(img_path)
                vis_path = join(opts.vis_dir, seq, img['name'][:-3] + 'jpg')

                bboxes_ltrb = bboxes.copy()
                if len(bboxes_ltrb):
                    bboxes_ltrb[:, 2:] += bboxes_ltrb[:, :2]
                if opts.overwrite or not isfile(vis_path):
                    vis_det(
                        I, bboxes_ltrb, labels,
                        class_names, masks if with_mask else None, scores,
                        out_scale=opts.vis_scale,
                        out_file=vis_path
                    )
    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}" --fps 30')

if __name__ == '__main__':
    main()