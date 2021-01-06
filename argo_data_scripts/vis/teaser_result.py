# Simulate the teaser result of an old video
import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import imread, parse_det_result, vis_det, eval_ccf
from track import vis_track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/data2/mengtial/ArgoVerse1.1/tracking')
    parser.add_argument('--annot-path', type=str, default='/data2/mengtial/ArgoVerse1.1/tracking/coco_fmt/val_c3.json')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--result-dir', type=str, default='/data2/mengtial/Exp/ArgoVerse1.1-c3-eta0/output/rt_mrcnn50_s0.5/val')
    parser.add_argument('--vis-dir', type=str, default='/data2/mengtial/Exp/ArgoVerse1.1-c3-eta0/special')
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--overwrite', action='store_true', default=True)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()
    mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    seq = 'f1008c18-e76e-3c24-adcc-da9858fac145'
    sid = seqs.index(seq)

    frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
    results = pickle.load(open(join(opts.result_dir, seq + '.pkl'), 'rb'))
    results_parsed = results['results_parsed']

    img = frame_list[153]
    result = results_parsed[15]
    bboxes, scores, labels, masks = result[:4]
    idx = [16]
    bboxes = bboxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    masks = masks[idx]

    img_path = join(opts.data_root, seq_dirs[sid], img['name'])
    I = imread(img_path).copy()
    vis_path = join(opts.vis_dir, 'teaser.jpg')
    if opts.overwrite or not isfile(vis_path):
        vis_det(
            I, bboxes, labels,
            class_names, masks, None,
            out_scale=opts.vis_scale,
            out_file=vis_path,
        )

if __name__ == '__main__':
    main()