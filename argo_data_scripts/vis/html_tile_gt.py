import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import imagetile

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse/tracking')
split = 'val'
annot_file = join(data_root, 'coco_fmt/htc_dconv2_ms_' + split + '_2.json')

out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse/vis'))
out_name = 'pgt2'
title = 'Pseudo GT with 9 classes'
metric = 'AP'
link_video = False
n_show = 120
n_consec = None
align = True        # align to the stride in each sequence
stride = 30
random.seed(1)

folder = join(data_root, 'coco_fmt/vis/htc_dconv2_ms_' + split + '_2')
print(f'python vis/make_videos.py "{folder}" --fps 30')

srv_dir = data_dir
srv_port = 40001

##

db = json.load(open(annot_file))
imgs = db['images']
seqs = db['sequences']


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
    seq = seqs[img['sid']]
    img_paths.append(join(seq, img['name'][:-3] + 'jpg'))
    vid_paths.append(seq + '.mp4')

paths = [join(folder, p) for p in img_paths]
if link_video:
    hrefs = [join(folder, p) for p in vid_paths]

imagetile(
    paths, 4,
    join(out_dir, out_name + '.html'),
    title,
    # href=links,
    href=hrefs if link_video else paths,
    pathrep=data_dir,
    imscale=0.2,
)

host_name = socket.gethostname()
url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)