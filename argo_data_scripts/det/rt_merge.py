# Merge real-time results for each sequence
# Optionally, visualize the output
# This script does not need to run in real-time
raise Exception('Not updated with new detection interface')
import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

import mmcv

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import vis_det, eval_ccf
from track import vis_track

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)

    print('Merging results')
    results_ccf = []
    in_time = 0
    miss = 0
    shifts = 0

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        results, timestamps, input_fidx, _ = pickle.load(open(join(opts.result_dir, seq + '.pkl'), 'rb'))
        
        tidx_p1 = 0
        for i, img in enumerate(frame_list):
            # pred, gt association by time
            if vis_out:
                img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                I = mmcv.imread(img_path)
                vis_path = join(opts.vis_dir, seq, img['name'][:-3] + 'jpg')

            t = (i + 1)/opts.fps
            while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
                tidx_p1 += 1
            if tidx_p1 == 0:
                # no output
                miss += 1
                if vis_out:
                    if opts.vis_scale != 1:
                        I = mmcv.imrescale(I, opts.vis_scale, interpolation='bilinear')
                    mmcv.imwrite(I, vis_path)
                continue
            
            tidx = tidx_p1 - 1
            result = results[tidx]
            ifidx = input_fidx[tidx]
            in_time += int(i == ifidx)
            shifts += i - ifidx

            bboxes = result['bboxes']
            scores = result['scores']
            labels = result['labels']
            masks = result['masks'] if 'masks' in result else None
            tracks = result['tracks'] if 'tracks' in result else None

            if vis_out:
                if opts.overwrite or not isfile(vis_path):
                    if tracks is None:
                        vis_det(
                            I, bboxes, labels,
                            class_names, masks, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path,
                            score_th=0,
                        )
                    else:
                        vis_track(
                            I, bboxes, tracks, labels,
                            class_names, masks, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path,
                            score_th=0,
                        )

            # convert to coco fmt
            bboxes_ltwh = bboxes.copy()
            if len(bboxes_ltwh):
                bboxes_ltwh[:, 2:] -= bboxes_ltwh[:, :2]
                bboxes_ltwh = bboxes_ltwh.tolist()

            for j in range(len(bboxes_ltwh)):
                result_dict = {
                    'image_id': img['id'],
                    'bbox': bboxes_ltwh[j],
                    'score': scores[j],
                    'category_id': labels[j],
                }
                if masks is not None:
                    result_dict['segmentation'] = masks[j]
                results_ccf.append(result_dict)

    out_path = join(opts.result_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    out_path = join(opts.result_dir, 'time_extra.txt')
    if opts.overwrite or not isfile(out_path):
        np.savetxt(out_path, [miss, in_time, shifts], fmt='%d')

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.result_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()