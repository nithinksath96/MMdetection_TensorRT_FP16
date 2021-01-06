import json, socket, pickle
from os.path import join, isfile, dirname, basename
from os import scandir

import numpy as np

from html4vision import Col, imagetable

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'
data_cfg = 'ArgoVerse1.1-debug'

out_dir = mkdir2(join(data_dir, 'Exp', data_cfg, 'vis'))
out_name = 'debug.html'
title = 'Debug'
metric = 'AP'
link_video = True

sel = 100

img_dir = '/data2/mengtial/Exp/ArgoVerse1.1-debug/pps_mrcnn50_nm_ds_s0.75_fba_iou_lin_pkt/5ab2697b-6e3e-3454-a36a-aba2c6f27818'


srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##
# print(f'python vis/make_videos.py "{img_dir}" --fps 30')


cols = [
    Col('id1', 'ID'),
    Col('img', 'Image', img_dir + '/*.jpg', sel),
]

imagetable(
    cols,
    join(out_dir, out_name),
    title,
    imscale=0.5,
    pathrep=srv_dir,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)