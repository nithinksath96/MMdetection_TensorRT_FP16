# Multiple detector at the same time, infinite GPUs
# + kf forecasting

import argparse, json, pickle
from os.path import join, isfile
from time import perf_counter
from copy import deepcopy

from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from util.bbox import ltrb2ltwh_, ltwh2ltrb, ltwh2cxywh, cxywh2ltwh
from det import imread, parse_det_result, eval_ccf, result_from_ccf
from track import vis_track
from track.bbox_filter import BboxFilterEx


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-root', type=str, required=True)
    # parser.add_argument('--annot-path', type=str, required=True)
    # # parser.add_argument('--no-mask', action='store_true', default=False)
    # parser.add_argument('--forecast', type=str, default='kf')
    # parser.add_argument('--forecast-before-assoc', action='store_true', default=False)
    # parser.add_argument('--fps', type=float, default=30)
    # parser.add_argument('--exp-dir', type=str, default=None)
    # parser.add_argument('--out-dir', type=str, required=True)
    # parser.add_argument('--vis-scale', type=float, default=None)
    # parser.add_argument('--vis-dir', type=str, default=None)
    # parser.add_argument('--no-eval', action='store_true', default=False)
    # parser.add_argument('--eval-mask', action='store_true', default=False)
    # parser.add_argument('--overwrite', action='store_true', default=False)

    parser.add_argument('--data-root', type=str, default='D:/Data')
    parser.add_argument('--annot-path', type=str, default='D:/Data/ArgoVerse1.1/tracking/coco_fmt/val.json')
    # parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--forecast', type=str, default='kf')
    parser.add_argument('--forecast-before-assoc', action='store_true', default=True)
    parser.add_argument('--fast-birth-tracks', action='store_true', default=False)
    parser.add_argument('--fast-kill-tracks', action='store_true', default=False)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--exp-dir', type=str, default='D:/Data/Exp')
    parser.add_argument('--out-dir', type=str, default='D:/Data/Exp/ArgoVerse1.1/output-inf-multi/debug/val')
    parser.add_argument('--vis-scale', type=float, default=None)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=True)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    # caches = [
    #     join(opts.exp_dir, 'ArgoVerse1.1-c3-eta0/output/retina50_s0.2/val/results_ccf.pkl'),
    #     join(opts.exp_dir, 'ArgoVerse1.1-c3-eta0/output/mrcnn50_nm_s0.5/val/results_ccf.pkl'),
    #     join(opts.exp_dir, 'ArgoVerse1.1-c3-eta0/output/mrcnn50_nm_s0.75/val/results_ccf.pkl'),
    # ]

    # # should be sorted
    # rtfs = [1, 2, 3]

    # caches = [
    #     join(opts.exp_dir, 'ArgoVerse1.1/output/mrcnn50_nm_th0_s0.75/val/results_ccf.pkl'),
    #     join(opts.exp_dir, 'ArgoVerse1.1/output/mrcnn50_nm_th0_s1.0/val/results_ccf.pkl'),
    # ]

    # # should be sorted
    # rtfs = [3, 5]

    caches = [
        join(opts.exp_dir, 'ArgoVerse1.1/output/mrcnn50_nm_th0_s0.75/val/results_ccf.pkl'),
        join(opts.exp_dir, 'ArgoVerse1.1/output/cmrcnn101_nm_s1.0/val/results_ccf.pkl'),
    ]

    # should be sorted
    rtfs = [3, 5]

    n_method = len(caches)
    max_history = max(rtfs)

    cache_ccfs = [
        pickle.load(open(path, 'rb'))
            for path in caches
    ]
    cache_end_idx = n_method*[0]

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]

    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    results_ccf = []

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        w_img, h_img = db.imgs[0]['width'], db.imgs[0]['height']
        bf = BboxFilterEx(
            w_img, h_img,
            forecast=opts.forecast,
            forecast_before_assoc=opts.forecast_before_assoc,
        )

        # backward in time
        # cur, cur - 1, ..., cur - max_history
        result_buffer = (max_history + 1)*[None]
        result_buffer_marker = (max_history + 1)*[None]
        min_rtf = min(rtfs)

        for ii, img in enumerate(frame_list):
            # fetch results
            for i in range(n_method):
                ifidx = ii - rtfs[i]
                if ifidx < 0:
                    break
                cache_end_idx[i], bboxes, scores, labels = \
                    result_from_ccf(cache_ccfs[i], frame_list[ifidx]['id'], cache_end_idx[i], mask=False)
                result_buffer[rtfs[i]] = (bboxes, scores, labels)
                result_buffer_marker[rtfs[i]] = i == 0
            if ii < min_rtf:
                # no method has finished yet
                bboxes, scores, labels, tracks = [], [], [], []
            else:
                s = max_history
                while s >= 0 and result_buffer[s] is None:
                    s -= 1
                # if first result is one step ahead
                t = ii - s
                birth_tracks = True
                kill_tracks = True
                if s == max_history:
                    bf.update(t, *result_buffer[s], None, birth_tracks, kill_tracks)
                    bf_this = deepcopy(bf)
                else:
                    bf_this = deepcopy(bf)
                    bf_this.update(t, *result_buffer[s], None, birth_tracks, kill_tracks)

                while 1:
                    # find next non-empty result
                    s -= 1
                    while s >= 0 and result_buffer[s] is None:
                        s -= 1
                    if s < 0:
                        break
                    if result_buffer_marker[s]:
                        birth_tracks = opts.fast_birth_tracks
                        kill_tracks = opts.fast_kill_tracks
                    else:
                        birth_tracks = True
                        kill_tracks = True

                    t = ii - s
                    bf_this.update(t, *result_buffer[s], None, birth_tracks, kill_tracks)
                
                bboxes, scores, labels, _, tracks = \
                    bf_this.predict(ii)

                # shift result buffer
                result_buffer.pop(max_history)
                result_buffer.insert(0, None)
                result_buffer_marker.pop(max_history)
                result_buffer_marker.insert(0, None)


            n = len(bboxes)
            for i in range(n):
                result_dict = {
                    'image_id': img['id'],
                    'bbox': bboxes[i],
                    'score': scores[i],
                    'category_id': labels[i],
                }
                # if masks is not None:
                #     result_dict['segmentation'] = masks[i]
                results_ccf.append(result_dict)

            if vis_out:
                img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                I = imread(img_path)
                vis_path = join(opts.vis_dir, seq, img['name'][:-3] + 'jpg')

                bboxes_ltrb = ltwh2ltrb(bboxes) if n else []
                if opts.overwrite or not isfile(vis_path):
                    vis_track(
                        I, bboxes_ltrb, tracks, labels,
                        class_names, None, scores,
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
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()