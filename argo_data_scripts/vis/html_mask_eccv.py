import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import imagetile

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'

out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse1.1/visf'))
out_name = 'mask_eccv'
title = 'Mask ECCV'

srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##

seq = 'cb762bb1-7ce1-3ba5-b53d-13c159b532c8'
frame_idx = 330

img_paths = [
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/htc_dconv2_ms_s1.0/val/{seq}/{frame_idx:06d}.jpg'),
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/srt_cmrcnn101_vm_s1.0/val/{seq}/{frame_idx:06d}.jpg'),
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_vm_ds_s0.75_fba_iou_lin_pkt/val/{seq}/{frame_idx:06d}.jpg'),
]

vid_paths = [
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/htc_dconv2_ms_s1.0/val/{seq}.mp4'),
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/srt_cmrcnn101_vm_s1.0/val/{seq}.mp4'),
    join(data_dir, f'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_vm_ds_s0.75_fba_iou_lin_pkt/val/{seq}.mp4'),
] 


captions = [
    'Offline 38.6',
    'Real-time 6.3',
    'Ours (Single GPU) 17.6',
]


# img_paths = [
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/vp_mask/5ab2697b-6e3e-3454-a36a-aba2c6f27818/000188.jpg'),
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/pss/5ab2697b-6e3e-3454-a36a-aba2c6f27818/000188.jpg'),
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/pss2/5ab2697b-6e3e-3454-a36a-aba2c6f27818/000188.jpg'),
# ]

# vid_paths = [
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/vp_mask/5ab2697b-6e3e-3454-a36a-aba2c6f27818.mp4'),
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/pss/5ab2697b-6e3e-3454-a36a-aba2c6f27818.mp4'),
#     join(data_dir, 'Exp/ArgoVerse1.1/vid/pss2/5ab2697b-6e3e-3454-a36a-aba2c6f27818.mp4'),
# ]


# captions = [
#     'Long Version',
#     'Short Version 1',
#     'Short Version 2',
# ]


imagetile(
    img_paths, 4,
    join(out_dir, out_name + '.html'),
    title,
    caption=captions,
    href=vid_paths,
    pathrep=data_dir,
    imscale=0.2,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}.html'
print(url)