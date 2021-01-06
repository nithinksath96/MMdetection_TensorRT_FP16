import csv, pickle
from glob import glob
from os.path import join, isfile, split

import numpy as np

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
prefix = join(data_dir, 'Exp/ArgoVerse')
out_dir = mkdir2(join(prefix, 'runtime-zoo', '1080ti'))

## Detection

# methods = [
#     *(4*['mrcnn_r50_no_mask']),
#     'retina_r50_no_mask',
# ]

# scales = [
#     *[0.25, 0.5, 0.75, 1],
#     '0.5',
# ]

methods = ['mrcnn_r50_no_mask_2']
scales = [0.5]


for m, s in zip(methods, scales):
    rt_info = pickle.load(open(join(prefix, f'output/rt_{m}/s{s}_val/time_all.pkl'), 'rb'))
    rt_samples = rt_info[0]
    rt_dist = {'type': 'empirical', 'samples': rt_samples}
    pickle.dump(rt_dist, open(join(out_dir, f'{m}_s{s}.pkl'), 'wb'))


# Detection as tracking
# methods = [
#     'dat_mrcnn50_nm_track_only_s0.5_d15_1',
# ]

# out_names = [
#     'dat_mrcnn50_nm_track_only_s0.5_1',
# ]

# for m, out_name in zip(methods, out_names):
#     rt_info = pickle.load(open(join(prefix, f'output/rt_{m}/val/time_info.pkl'), 'rb'))
#     which_cfg_all = np.array(rt_info['which_cfg_all'])
#     rt_samples = np.array(rt_info['runtime_all'])[which_cfg_all == 1]
#     rt_samples = rt_samples[rt_samples > 0.002] # ignore zero detection trivial cases
#     rt_dist = {'type': 'empirical', 'samples': rt_samples.tolist()}
#     pickle.dump(rt_dist, open(join(out_dir, f'{out_name}.pkl'), 'wb'))


## CF-based tracking

