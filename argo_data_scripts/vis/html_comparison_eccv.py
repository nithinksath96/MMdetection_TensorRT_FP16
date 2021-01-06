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
data_cfg = 'ArgoVerse1.1'

out_dir = mkdir2(join(data_dir, 'Exp', data_cfg, 'visf'))
out_name = 'eccv.html'
title = 'ECCV'
metric = 'AP'
link_video = True
n_show = 2
np.random.seed(3)

names = [
    # 'A: Offline<br>B: Real-Time',
    # 'A: Det Accurate<br>B: Det Fast',
    'A: Det Fast (CPU-pre)<br>B: Det Optimized',
    'A: Det Optimized<br>B: Det Optimized + Scheduling + Forecasting',
    'A: Det Optimized + Scheduling + Forecasting<br>B: + Infinite GPUs',
    # 'A: Det + Visual Tracking + Forecasting<br>B: Det + Visual Tracking (simulated x2) + Forecasting',
    # 'A: Offline SOTA<br>B: Best Streaming (Single GPU)',
    # 'A: Offline SOTA<br>B: Best Streaming (Inf GPUs)',

]

dirs = [
    # join(data_dir, 'Exp', data_cfg, 'visf', 'htc_dconv2_ms_nm_s1.0-vs-rt_htc_dconv2_ms_cpupre_s1', split),
    # join(data_dir, 'Exp', data_cfg, 'visf', 'rt_htc_dconv2_ms_cpupre_s1-vs-rt_retina50_s0.2_pkt', split),
    join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'rt_retina50_cpupre_s0.2_pkt-vs-rt_mrcnn50_nm_s0.5_pkt', split),
    join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'rt_mrcnn50_nm_s0.5_pkt-vs-pps_mrcnn50_nm_ds_s0.75_kf_fba_iou_lin_pkt', split),
    join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'pps_mrcnn50_nm_ds_s0.75_kf_fba_iou_lin_pkt-vs-pps_mrcnn50_nm_inf_s0.75_kf_fba_iou_lin_pkt', split),
    # join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'pps_mrcnn50_nm_ds_s0.75_fba_iou_lin_pkt-vs-pps_mrcnn50_nm_inf_s0.75_3_stitched_pkt', split),
    # join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'pps_dat_mrcnn50_nm_d5_s0.75_pkt-vs-pps_dat_mrcnn50_nm_d15_track-pf-2_s0.75_pkt', split),
    # join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'htc_dconv2_ms_nm_s1.0-vs-pps_mrcnn50_nm_ds_s0.75_fba_iou_lin_pkt', split),
    # join(data_dir, 'Exp', data_cfg, 'visf-th0.5', 'htc_dconv2_ms_nm_s1.0-vs-pps_mrcnn50_nm_inf_s0.75_3_stitched_pkt', split),
]

frame_indices = [
    # [477, 35],
    # [477, 35],
    [477, 35],
    [508, 71],
    [508, 71],
    # [508, 71],
    # [508, 71],
    # [508, 71],
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

# sel = np.random.choice(n_seq, n_show, replace=False)
# sel = list(range(n_show))
sel = [1, 0]

n = len(names)
img_paths = [[] for i in range(n)]
vid_paths = [[] for i in range(n)]

for i in range(n):
    for j, idx in enumerate(sel):
        seq = seqs[idx]
        frames = [item.name for item in scandir(join(dirs[0], seq)) if item.is_file() and item.name.endswith('.jpg')]
        frames = sorted(frames)
        print(len(frames), join(dirs[0], seq))
        # select a random frame at the beginning for this sequence
        # 
        # if i == 0:
        # if i < 3 or j == 1:
        sel_idx = frame_indices[i][j]
        # else:
            # sel_idx = np.random.choice(len(frames))
            # print(sel_idx)
        print(sel_idx)
        frame = frames[sel_idx]
        img_paths[i].append(join(seq, frame))
        vid_paths[i].append(seq + '.mp4')

cols = []
summary_row = []

for i, name in enumerate(names):
    paths = [join(dirs[i], p) for p in img_paths[i]]
    if link_video:
        hrefs = [join(dirs[i], p) for p in vid_paths[i]]
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
    imscale=0.3,
    # style='body {margin: 0}\n.html4vision td img {display: block; margin-left: auto; margin-right: auto;}',
    style='img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}',
    pathrep=srv_dir,
    copyright=False,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)