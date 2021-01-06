import json, socket, pickle
from os.path import join, isfile, dirname, basename
from os import scandir

import numpy as np

from html4vision import Col, imagetile


import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'
data_cfg = 'ArgoVerse1.1'

out_dir = mkdir2(join(data_dir, 'Exp', data_cfg, 'visf'))
out_name = 'best_single_all_seq_mask.html'
title = 'Best Single GPU (Mask)'
link_video = True
n_show = 100
np.random.seed(0)

folder = join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'pps_mrcnn50_ds_s0.5_kf_fba_iou_lin_pkt', split)
print(f'python vis/make_videos.py "{folder}" --fps 30')

srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##

seqs = sorted([item.name for item in scandir(folder) if item.is_dir()])
n_seq = len(seqs)



img_paths = []
vid_paths = []

for i, seq in enumerate(seqs):
    frames = [item.name for item in scandir(join(folder, seq)) if item.is_file() and item.name.endswith('.jpg')]
    frames = sorted(frames)
    frame = np.random.choice(frames)
    img_paths.append(join(folder, seq, frame))
    vid_paths.append(join(folder, seq + '.mp4'))

hrefs = vid_paths if link_video else img_paths
captions = [f'{i+1}. {seq}' for i, seq in enumerate(seqs)]

imagetile(
    img_paths, 3,
    join(out_dir, out_name),
    title,
    caption=captions,
    href=hrefs,
    subset=n_show,
    imscale=0.3,
    pathrep=srv_dir,
    copyright=False,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)