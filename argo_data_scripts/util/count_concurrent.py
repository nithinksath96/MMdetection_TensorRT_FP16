# given rt_ series input, count the max number of concurrent jobs
# current implementation only applies to inf results, where processing starts immediately

import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--type', type=str, default='det')
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    out_dir = mkdir2(opts.out_dir) if opts.out_dir else opts.result_dir

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    n_concurrent = []

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        results = pickle.load(open(join(opts.result_dir, seq + '.pkl'), 'rb'))
        # use raw results when possible in case we change class subset during evaluation
        if opts.type == 'det':
            timestamps = results['timestamps']
            input_fidx = results['input_fidx']
        else:
            det1_timestamps = results['det1_timestamps']
            det2_timestamps = results['det2_timestamps']
            det1_input_fidx = results['det1_input_fidx']
            det2_input_fidx = results['det2_input_fidx']
            timestamps = np.concatenate((det1_timestamps, det2_timestamps))
            input_fidx = np.concatenate((det1_input_fidx, det2_input_fidx))

        t_start = np.asarray(input_fidx)/opts.fps
        t_end = np.asarray(timestamps)
        t_all = np.concatenate((t_start, t_end))
        order = np.argsort(t_all)

        n_output = len(t_start)
        n_current = 0
        max_current = 0
        for i in order:
            if i < n_output:
                # start
                n_current += 1
                max_current = max(max_current, n_current)
            else:
                # end
                n_current -= 1
        n_concurrent.append(max_current)

    print(f'Max number of concurrent jobs {max(n_concurrent)}')

    out_path = join(out_dir, 'n_concurrent.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(n_concurrent, open(out_path, 'wb'))

if __name__ == '__main__':
    main()