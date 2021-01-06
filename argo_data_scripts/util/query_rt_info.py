import pickle
from os.path import join

import numpy as np

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import print_stats


file_path = '/data2/mengtial/Exp/ArgoVerse1.1/output/rt_htc_dconv2_ms_nm_s1.0/val/time_info.pkl'
time_info = pickle.load(open(file_path, 'rb'))
runtime_all_np = np.array(time_info['runtime_all'])

s2ms = lambda x: 1e3*x
# print_stats(det1_runtime, 'det1 (ms)', cvt=s2ms)
# print_stats(det2_runtime, 'det2 (ms)', cvt=s2ms)
print_stats(runtime_all_np, 'overall (ms)', cvt=s2ms)

# file_path = '/data2/mengtial/Exp/ArgoVerse-pgt-gpupre/output/rt_dat_mrcnn50_nm_d15_s0.5/val/time_info.pkl'
# time_info = pickle.load(open(file_path, 'rb'))
# runtime_all_np = np.array(time_info['runtime_all'])
# which_cfg = np.array(time_info['which_cfg_all'], np.bool)
# det1_runtime = runtime_all_np[np.logical_not(which_cfg)]
# det2_runtime = runtime_all_np[which_cfg]

# s2ms = lambda x: 1e3*x
# print_stats(det1_runtime, 'det1 (ms)', cvt=s2ms)
# print_stats(det2_runtime, 'det2 (ms)', cvt=s2ms)
# print_stats(runtime_all_np, 'overall (ms)', cvt=s2ms)



# print('Init (ms): mean: %g; std: %g; min: %g; max: %g' % (
#     runtime_all_np.mean(),
#     runtime_all_np.std(ddof=1),
#     runtime_all_np.min(),
#     runtime_all_np.max(),
# ))

# aa = pickle.load(open('/data2/mengtial/Exp/ArgoVerse/output/cv2csrdcf_mrcnn50_d10/s1_val/time_tracker.pkl', 'rb'))
# runtime_all_np = 1e3*np.array(aa['rt_tracker_init'])[100:10101]
# print(len(runtime_all_np))
# print('Init (ms): mean: %g; std: %g; min: %g; max: %g' % (
#     runtime_all_np.mean(),
#     runtime_all_np.std(ddof=1),
#     runtime_all_np.min(),
#     runtime_all_np.max(),
# ))
# runtime_all_np = 1e3*np.array(aa['rt_tracker_update'])[100:10101]
# print('Update (ms): mean: %g; std: %g; min: %g; max: %g' % (
#     runtime_all_np.mean(),
#     runtime_all_np.std(ddof=1),
#     runtime_all_np.min(),
#     runtime_all_np.max(),
# ))

# status = np.array(aa['tracker_status'])
# print('%d; %g' % (len(status), status.mean()))

