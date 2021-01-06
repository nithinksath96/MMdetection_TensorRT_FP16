import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import Col, imagetable

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root_1 = join(data_dir, 'ArgoVerse/tracking')
data_root_2 = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'
annot_file = join(data_root_1, 'coco_fmt/htc_dconv2_ms_' + split + '.json')
video_dir = 'vid_ring_front_center'

out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse/vis'))
out_name = 'av1.0-vs-av1.1.html'
title = 'AV1.0-vs-AV1.1'
link_video = True
n_show = 90
n_consec = 6
align = False        # align to the stride in each sequence
stride = 30
random.seed(2)

names = [
    'ArgoVerse 1.0',
    'ArgoVerse 1.1',
]

dirs = [data_root_1, data_root_2]


srv_dir = data_dir
srv_port = 40001
host_name = 'trinity.vision.cs.cmu.edu'

##

db = json.load(open(annot_file))
imgs = db['images']
seqs = db['sequences']
seq_dirs = db['seq_dirs']


n_img = len(imgs)

if n_consec is None:
    sel = random.choices(list(range(n_img)), k=n_show)
elif align:   
    start_idx = []
    last_sid = None
    for i, img in enumerate(imgs):
        if img['sid'] != last_sid:
            start_idx.append(i)
            last_sid = img['sid']
    start_idx = np.array(start_idx)

    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    sel = np.array(sel)
    sel *= n_consec
    for i in range(len(sel)):
        diff = sel[i] - start_idx
        diff[diff < 0] = n_img
        nearest = np.argmin(diff)
        sel[i] -= (sel[i] - start_idx[nearest]) % stride
    # it is possible to have duplicated sel, but ignore for now
    consecs = np.arange(n_consec)
    sel = [i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()
else:
    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    consecs = np.arange(n_consec)
    sel = [n_consec*i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()

# sel = list(range(n_show))

img_paths = []
vid_paths = []

for idx in sel:
    img = imgs[idx]
    sid = img['sid']
    seq = seqs[sid]
    img_paths.append(join(seq_dirs[sid], img['name']))
    vid_paths.append(join(video_dir, split, seq + '.mp4'))

cols = [Col('id1', 'ID')]

for i, name in enumerate(names):
    paths = [join(dirs[i], p) for p in img_paths]
    if link_video:
        hrefs = [join(dirs[i], p) for p in vid_paths]
    else:
        hrefs = paths
    cols.append(
        Col('img', name, paths, href=hrefs)
    )


imagetable(
    cols,
    join(out_dir, out_name),
    title,
    imscale=0.5,
    sortable=True,
    sticky_header=True,
    sort_style='materialize',
    style='body {margin: 0}',
    pathrep=srv_dir,
)

if host_name is None:
    host_name = socket.gethostname()
url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)