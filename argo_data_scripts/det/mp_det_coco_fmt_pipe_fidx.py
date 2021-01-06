# multi-processing version of rt det, only passing frame index and results between processes
import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import traceback

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from det import imread, parse_det_result
from det.det_apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--det-stride', type=float, default=1)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(opts, frame_recv, det_res_send):
    try:
        model = init_detector(opts)

        # warm up the GPU
        w_img, h_img = 1200, 1920
        _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
        torch.cuda.synchronize()

        while 1:
            fidx = frame_recv.recv()
            if type(fidx) is list:
                # new video, read all images in advance
                frame_list = fidx
                frames = [imread(img_path) for img_path in frame_list]
                # signal ready, no errors
                det_res_send.send(None)
                continue
            elif fidx is None:
                # exit flag
                break
            fidx, t1 = fidx
            img = frames[fidx]
            t2 = perf_counter() 
            t_send_frame = t2 - t1

            result = inference_detector(model, img)
            torch.cuda.synchronize()

            t3 = perf_counter()
            det_res_send.send([result, t_send_frame, t3])

    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        det_res_send.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mp.set_start_method('spawn')
    frame_recv, frame_send = mp.Pipe(False)
    det_res_recv, det_res_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, frame_recv, det_res_send))
    det_proc.start()

    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    runtime_all = []
    n_processed = 0
    n_total = 0

    t_send_frame_all = []
    t_recv_res_all = []

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        frame_list = [join(opts.data_root, seq_dirs[sid], img['name']) for img in frame_list]
        n_frame = len(frame_list)
        n_total += n_frame
        
        timestamps = []
        results_raw = []
        results_parsed = []
        input_fidx = []
        runtime = []
        last_fidx = None
        stride_cnt = 0
        
        # let detector process to read all the frames
        frame_send.send(frame_list)
        init_error = det_res_recv.recv() # wait till the detector is ready
        if init_error is not None:
            raise init_error

        t_total = n_frame/opts.fps
        t_start = perf_counter()
        while 1:
            t1 = perf_counter()
            t_elapsed = t1 - t_start
            if t_elapsed >= t_total:
                break

            # identify latest available frame
            fidx = int(np.floor(t_elapsed*opts.fps))
            #   t_elapsed/t_total *n_frame
            # = t_elapsed*opts.fps
            if fidx == last_fidx:
                continue
            
            last_fidx = fidx
            if stride_cnt % opts.det_stride == 0:
                stride_cnt = 1
            else:
                stride_cnt += 1
                continue

            t_start_frame = perf_counter()
            frame_send.send((fidx, t_start_frame))
            result = det_res_recv.recv() # wait
            if isinstance(result, Exception):
                raise result
            result, t_send_frame, t_start_res = result
            bboxes, scores, labels, masks = \
                parse_det_result(result, coco_mapping, n_class)

            t2 = perf_counter()
            t_send_frame_all.append(t_send_frame)
            t_recv_res_all.append(t2 - t_start_res)
            t_elapsed = t2 - t_start
            if t_elapsed >= t_total:
                break
            
            timestamps.append(t_elapsed)
            results_raw.append(result)
            results_parsed.append((bboxes, scores, labels, masks))
            
            input_fidx.append(fidx)
            runtime.append(t2 - t1)

        out_path = join(opts.out_dir, seq + '.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump({
                'results_raw': results_raw,
                'results_parsed': results_parsed,
                'timestamps': timestamps,
                'input_fidx': input_fidx,
                'runtime': runtime,
            }, open(out_path, 'wb'))

        runtime_all += runtime
        n_processed += len(results_raw)

    # terminates the child process
    frame_send.send(None)

    runtime_all_np = np.asarray(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/opts.fps).sum()

    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'runtime_all': runtime_all,
            'n_processed': n_processed,
            'n_total': n_total,
            'n_small_runtime': n_small_runtime,
        }, open(out_path, 'wb'))

    # convert to ms for display
    s2ms = lambda x: 1e3*x

    print(f'{n_processed}/{n_total} frames processed')
    print_stats(runtime_all_np, 'Runtime (ms)', cvt=s2ms)
    print(f'Runtime smaller than unit time interval: '
        f'{n_small_runtime}/{n_processed} '
        f'({100.0*n_small_runtime/n_processed:.4g}%)')
    print_stats(t_send_frame_all, 'Time spent on sending the frame (ms)', cvt=s2ms)
    print_stats(t_recv_res_all, 'Time spent on receiving the result (ms)', cvt=s2ms)

if __name__ == '__main__':
    main()