import csv, pickle
from glob import glob
from os.path import join, isfile, split

import numpy as np

work_dir = '/data2/mengtial/Exp/ArgoVerse-pgt-gpupre/output'
data_split = 'val'

header = [
    'Method',
    'AP', 'AP-0.5', 'AP-0.7',
    'AP-small', 'AP-medium', 'AP-large',
    'RT mean', 'RT std', 'RT min', 'RT max',
    'Small RT', 'Miss', 'In-time', 'Shifts',
]

runs = glob(join(work_dir, '*', data_split))

out_path = join(work_dir, 'all_methods.csv')
with open(out_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in runs:
        eval_path = join(r, 'eval_summary.pkl')
        if not isfile(eval_path):
            continue
        eval_summary = pickle.load(open(eval_path, 'rb'))['stats'][:6]

        name = split(split(r)[0])[1]

        time_path = join(r, 'time_all.pkl')
        # (runtime_all, n_processed, n_total, n_small_runtime)
        if isfile(time_path):
            time_all = pickle.load(open(time_path, 'rb'))
            runtime = np.array(time_all[0])
            time_stats = 1e3*np.array([
                runtime.mean(),
                runtime.std(ddof=1),
                runtime.min(),
                runtime.max(),
            ])
            np.append(time_stats, time_all[3]/time_all[1])
            time_extra_path = join(r, 'time_extra.txt')
            # if isfile(time_extra_path):
            #     time_extra = np.loadtxt(time_extra_path)
            #     time_stats = np.append(time_stats, time_extra[0])
            #     time_stats = np.concatenate((time_stats, time_extra[1:]/time_all[2]))
            time_stats = time_stats.tolist()
        else:
            time_stats = []

        w.writerow([name] + eval_summary.tolist() + time_stats)

print(out_path)
