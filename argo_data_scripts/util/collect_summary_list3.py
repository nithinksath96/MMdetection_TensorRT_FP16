import csv, pickle
from glob import glob
from os.path import join, isfile, split

import numpy as np

exp_list = [
    'frcnn_ft_e3_lr2e-2_ts0.5_bf_none',
    'frcnn_ft_e3_lr2e-2_ts0.5_bf_fba_iou_lin',
    'frcnn_ft_e3_lr2e-2_ts0.5_bf_kf',
    'srt_f2f',
    'srt_f2f_ds',
    'srt_f2f_rt3f',
    'srt_f2f_rt3f_ds',
    'pps_f2f_rt3f_ds_kf',
    'srt_f2f_rt3f_inf',
    'pps_f2f_rt3f_inf_kf',
]
summary_name = 'f2f'
work_dir = '/data2/mengtial/Exp/ArgoVerse1.1/output-train'
data_split = 'val'
mask = False

header = [
    'Method',
    'AP', 'AP-0.5', 'AP-0.7',
    'AP-small', 'AP-medium', 'AP-large',
    'RT mean', 'RT std', 'RT min', 'RT max',
    'Small RT', 'Miss', 'In-time', 'Mismatch',
]


if isinstance(exp_list, str):
    runs = open(exp_list).readlines()
    runs = [r.strip() for r in runs]
else:
    runs = exp_list

n_method = 0
out_path = join(work_dir, f'{summary_name}.csv')
with open(out_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(header)
    for name in runs:
        result_dir = join(work_dir, name, data_split)
        eval_path = join(result_dir, 'eval_summary_mask.pkl' if mask else 'eval_summary.pkl')
        if not isfile(eval_path):
            continue
        n_method += 1
        eval_summary = 100*pickle.load(open(eval_path, 'rb'))['stats'][:6]

        time_path = join(result_dir, 'time_info.pkl')
        if isfile(time_path):
            time_info = pickle.load(open(time_path, 'rb'))
            n_total = time_info['n_total']
            if 'runtime_all' in time_info:
                runtime_all = 1e3*np.asarray(time_info['runtime_all'])
                n_processed = time_info['n_processed']
                n_small_runtime = time_info['n_small_runtime']
                time_stats = [
                    runtime_all.mean(),
                    runtime_all.std(ddof=1),
                    runtime_all.min(),
                    runtime_all.max(),
                    n_small_runtime/n_processed,
                ]
            else:
                time_stats = 5*['']
            time_extra_path = join(result_dir, 'eval_assoc.pkl')
            if isfile(time_extra_path):
                time_extra = pickle.load(open(time_extra_path, 'rb'))
                miss = time_extra['miss']
                in_time = time_extra['in_time']
                mismatch = time_extra['mismatch']
                time_stats.append(miss)
                if 'n_processed' in time_info:
                    time_stats += [
                        in_time/n_processed, mismatch/n_processed
                    ]
        else:
            time_stats = []

        w.writerow([name] + eval_summary.tolist() + time_stats)

print(f'{n_method} methods found')
print(out_path)
