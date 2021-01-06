import argparse, pickle
from glob import glob
from os.path import join, isfile, dirname

import numpy as np

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-info', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--method-type', type=str, default='det')
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()
    if not opts.overwrite and isfile(opts.out_path):
        return
    mkdir2(dirname(opts.out_path))

    time_info = pickle.load(open(opts.time_info, 'rb'))
    if opts.method_type == 'det':
        rt_samples = time_info['runtime_all']
        rt_dist = {'type': 'empirical', 'samples': rt_samples}
    elif opts.method_type == 'dat':
        which_cfg_all = np.array(time_info['which_cfg_all'])
        rt_samples = np.array(time_info['runtime_all'])[which_cfg_all == 1]
        rt_samples = rt_samples[rt_samples > 0.002] # ignore zero detection trivial cases
        rt_dist = {'type': 'empirical', 'samples': rt_samples.tolist()}
    else:
        raise ValueError(f'Unknown method type "{opts.method_type}"')
    pickle.dump(rt_dist, open(opts.out_path, 'wb'))

if __name__ == '__main__':
    main()
