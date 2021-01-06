import json
from os.path import join
from glob import glob
from tqdm import tqdm

import numpy as np

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import vis_det, imread
from track import vis_track

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'
annot_file = join(data_root, 'coco_fmt/' + split + '_track.json')
show_track = True

out_dir = mkdir2(join(data_root, 'coco_fmt/vis/' + split + '_track'))
out_scale = 1

##

# not assuming consecutive storage imgs and annots (obj, frame, seq)
dataset = json.load(open(annot_file))
cats = dataset['categories']
imgs = dataset['images']
annots = dataset['annotations']
seqs = dataset['sequences']
seq_dirs = dataset['seq_dirs']

class_names = [c['name'] for c in cats]

last_sid = None
aidx = 0
for img in tqdm(imgs):
    img_name = img['name']
    iid = img['id']
    sid = img['sid']

    if sid != last_sid:
        in_dir_seq = join(data_root, seq_dirs[sid])
        out_dir_seq = join(out_dir, seqs[sid])
        last_sid = sid

    I = imread(join(in_dir_seq, img['name']))

    bboxes = []
    masks = []
    labels = []
    if show_track:
        tracks = []

    while aidx < len(annots):
        ann = annots[aidx]
        if ann['image_id'] != iid:
            break
        bboxes.append(ann['bbox'])
        if 'segmentation' in ann:
            masks.append(ann['segmentation'])
        else:
            masks = None
        labels.append(ann['category_id'])
        if show_track:
            tracks.append(ann['track'])
        aidx += 1

    if bboxes:
        bboxes = np.array(bboxes)
        bboxes[:, 2:] += bboxes[:, :2]

    out_path = join(out_dir_seq, img_name[:-3] + 'jpg')
    if show_track:
        vis_track(
            I, bboxes, tracks, labels,
            class_names, masks,
            out_scale=out_scale,
            out_file=out_path
        )
    else:  
        vis_det(
            I, bboxes, labels,
            class_names, masks,
            out_scale=out_scale,
            out_file=out_path
        )

