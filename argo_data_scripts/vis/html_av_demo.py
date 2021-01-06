import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import Col, imagetable

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse/tracking')
split = 'val'
annot_file = join(data_root, 'coco_fmt/htc_dconv2_ms_' + split + '.json')

out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse/vis'))
out_name = 'dat_demo.html'
title = 'Dat Demo'
metric = 'AP'
link_video = False
n_show = 90
n_consec = 30
align = True        # align to the stride in each sequence
stride = 30
random.seed(1)


names = [
    'Annotation',
    'DAT track only D3',
]

dirs = [
    join(data_root, 'coco_fmt/vis/htc_dconv2_ms_' + split),
    join(data_dir, 'Exp/ArgoVerse/vis/srt_dat_mrcnn50_nm_track_only_s0.5_det2pf1.67_d3/' + split),
]

assert len(names) == len(dirs)

for d in dirs:
    print(f'python vis/make_videos.py "{d}" --fps 30')

srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##

db = json.load(open(annot_file))
imgs = db['images']
seqs = db['sequences']
imgs = [img for img in imgs if img['sid'] == 2]

n_img = len(imgs)

# if n_consec is None:
#     sel = random.choices(list(range(n_img)), k=n_show)
# elif align:   
#     start_idx = []
#     last_sid = None
#     for i, img in enumerate(imgs):
#         if img['sid'] != last_sid:
#             start_idx.append(i)
#             last_sid = img['sid']
#     start_idx = np.array(start_idx)

#     sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
#     sel = np.array(sel)
#     sel *= n_consec
#     for i in range(len(sel)):
#         diff = sel[i] - start_idx
#         diff[diff < 0] = n_img
#         nearest = np.argmin(diff)
#         sel[i] -= (sel[i] - start_idx[nearest]) % stride
#     # it is possible to have duplicated sel, but ignore for now
#     consecs = np.arange(n_consec)
#     sel = [i + consecs for i in sel]
#     sel = np.array(sel).flatten().tolist()
# else:
#     sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
#     consecs = np.arange(n_consec)
#     sel = [n_consec*i + consecs for i in sel]
#     sel = np.array(sel).flatten().tolist()

sel = list(range(n_show))

img_paths = []
vid_paths = []

for idx in sel:
    img = imgs[idx]
    seq = seqs[img['sid']]
    img_paths.append(join(seq, img['name'][:-3] + 'jpg'))
    vid_paths.append(seq + '.mp4')

cols = [Col('id1', 'ID')]
summary_row = [metric]

for i, name in enumerate(names):
    paths = [join(dirs[i], p) for p in img_paths]
    if link_video:
        hrefs = [join(dirs[i], p) for p in vid_paths]
    else:
        hrefs = paths
    cols.append(
        Col('img', name, paths, href=hrefs)
    )

    if 'Exp' in dirs[i]:
        eval_path = join(dirs[i].replace('/vis/', '/output/'), 'eval_summary.pkl')
        if isfile(eval_path):
            eval_summary = pickle.load(open(eval_path, 'rb'))
            summary_row.append('%.3g' % eval_summary['stats'][0])
        else:
            summary_row.append('')
    else:
        summary_row.append('')

imagetable(
    cols,
    join(out_dir, out_name),
    title,
    summary_row=summary_row,
    # imscale=0.3 + imsize=1
    imsize=(288, 180),
    sortable=True,
    sticky_header=True,
    sort_style='materialize',
    style='body {margin: 0}',
    pathrep=srv_dir,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)