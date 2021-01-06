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
data_cfg = 'ArgoVerse1.1-c3-eta0'

out_dir = mkdir2(join(data_dir, 'Exp', data_cfg, 'visf'))
out_name = 'compare.html'
title = 'Result Comparison'
metric = 'AP'
link_video = True
n_show = 2
np.random.seed(3)

names = [
    'A: Offline<br>B: Real-Time',
    'A: Det Accurate<br>B: Det Fast',
    'A: Det Fast<br>B: Det Optimized + Scheduling',
    'A: Det Optimized<br>B: Det Optimized + Scheduling + Forecasting',
    'A: Det Optimized + Scheduling + Forecasting<br>B: Det + Infinite GPUs + Forecasting',
    'A: Det + Visual Tracking + Forecasting<br>B: Det + Visual Tracking (simulated x2) + Forecasting',
]

dirs = [
    join(data_dir, 'Exp', data_cfg, 'visf', 'htc_dconv2_ms_nm_s1.0-vs-rt_htc_dconv2_ms_cpupre_s1', split),
    join(data_dir, 'Exp', data_cfg, 'visf', 'rt_htc_dconv2_ms_cpupre_s1-vs-rt_retina50_s0.2', split),
    join(data_dir, 'Exp', data_cfg, 'visf', 'rt_retina50_s0.2-vs-srt_mrcnn50_nm_ds_s0.5', split),
    join(data_dir, 'Exp', data_cfg, 'visf', 'srt_mrcnn50_nm_ds_s0.5-vs-pps_mrcnn50_nm_ds_s0.75_fba_iou_lin', split),
    join(data_dir, 'Exp', data_cfg, 'visf', 'pps_mrcnn50_nm_ds_s0.75_fba_iou_lin-vs-pps_mrcnn50_nm_inf_s0.75_3_stitched', split),
    join(data_dir, 'Exp', data_cfg, 'visf', 'pps_dat_mrcnn50_nm_d5_s0.75-vs-pps_dat_mrcnn50_nm_d15_track-pf-2_s0.75', split),
]

for d in dirs:
    print(f'python vis/make_videos.py "{d}" --fps 30')

srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##

seqs = sorted([item.name for item in scandir(dirs[0]) if item.is_dir()])
n_seq = len(seqs)

sel = np.random.choice(n_seq, n_show, replace=False)
# sel = list(range(n_show))

img_paths = []
vid_paths = []

for idx in sel:
    seq = seqs[idx]
    frames = [item.name for item in scandir(join(dirs[0], seq)) if item.is_file() and item.name.endswith('.jpg')]
    # select a random frame at the beginning for this sequence
    frame = np.random.choice(frames[:100])
    img_paths.append(join(seq, frame))
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

    method_names = basename(dirname(dirs[i]))
    idx = method_names.find('-vs-')
    assert idx >= 0
    method_A = method_names[:idx]
    method_B = method_names[idx + 4:]

    if 'Exp' in dirs[i]:
        eval_A = join(data_dir, 'Exp', data_cfg, 'output', method_A, split, 'eval_summary.pkl')
        eval_B = join(data_dir, 'Exp', data_cfg, 'output', method_B, split, 'eval_summary.pkl')
        if isfile(eval_A):
            eval_summary = pickle.load(open(eval_A, 'rb'))
            summary_A = '%.1f' % (100*eval_summary['stats'][0])
        else:
            summary_A = ''
        if isfile(eval_B):
            eval_summary = pickle.load(open(eval_B, 'rb'))
            summary_B = '%.1f' % (100*eval_summary['stats'][0])
        else:
            summary_B = ''
        summary_row.append(f'{summary_A} vs {summary_B}')
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
    style='body {margin: 0}\n.html4vision td img {display: block; margin-left: auto; margin-right: auto;}',
    pathrep=srv_dir,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)