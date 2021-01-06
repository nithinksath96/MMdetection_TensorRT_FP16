# Run detection in real-time setting on a COCO-format dataset
# import os
# os.system('taskset -p 0xffffffff %d' % os.getpid())

import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import traceback

from tqdm import tqdm
import numpy as np

import torch
import mmcv

from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from dbcode.dbinfo import coco2av, coco2kmots, kmots_classes, av_classes

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

    # parser.add_argument('--config', type=str, default='../mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    # parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='/data/mengtial/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(opts, frame_queue, res_queue):
    try:
        config = mmcv.Config.fromfile(opts.config)
        if opts.in_scale is not None:
            config.data.test.img_scale = opts.in_scale
            # mainly for SSD
            config.data.test.resize_keep_ratio = True
        if opts.no_mask:
            if 'mask_head' in config.model:
                config.model['mask_head'] = None

        model = init_detector(config, opts.weights)
        model.eval()

        # warm up the GPU
        _ = inference_detector(model, np.zeros((1200, 1920, 3), np.uint8))
        torch.cuda.synchronize()

        # signal ready, no errors
        res_queue.put(None)

        while 1:
            frame = frame_queue.get()
            if frame is None:
                # exit flag
                break
            frame, t1 = frame
            t2 = perf_counter() 
            t_send_frame = t2 - t1

            result = inference_detector(model, frame)
            torch.cuda.synchronize()

            t3 = perf_counter()
            # print('send: %.3g det: %.3g' % (t_send_frame, t3 - t2))

            res_queue.put([result, t_send_frame, t3])

    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        res_queue.put(Exception("".join(traceback.format_exception(*sys.exc_info()))))

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    # CUDA runtime does not support the fork
    mp.set_start_method('spawn')
    frame_queue = mp.Queue()
    res_queue = mp.Queue()
    det_proc = mp.Process(target=det_process, args=(opts, frame_queue, res_queue))
    det_proc.start()

    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    runtime_all = []
    n_processed = 0
    n_total = 0

    t_send_frame_all = 0
    t_recv_res_all = 0

    init_error = res_queue.get() # wait till the detector is ready
    if init_error is not None:
        raise init_error

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        # load all frames in advance
        frames = []
        for img in frame_list:
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            frames.append(mmcv.imread(img_path))
        n_frame = len(frames)
        n_total += n_frame
        
        timestamps = []
        results = []
        input_fidx = []
        runtime = []
        last_fidx = None
        stride_cnt = 0
        
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

            frame = frames[fidx]

            t_start_frame = perf_counter()

            frame_queue.put((frame, t_start_frame))
            result = res_queue.get() # wait
            if isinstance(result, Exception):
                raise result
            result, t_send_frame, t_start_res = result

            t2 = perf_counter()

            t_send_frame_all += t_send_frame
            t_recv_res_all += t2 - t_start_res
            # print('recv: %.3g elapsed: %.3g' % (t2 - t_start_res, t2 - t1))
            
            t_elapsed = t2 - t_start


            if t_elapsed >= t_total:
                break
            
            timestamps.append(t_elapsed)
            results.append(result)
            input_fidx.append(fidx)
            runtime.append(t2 - t1)

        out_path = join(opts.out_dir, seq + '.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump((
                results, timestamps,
                input_fidx, runtime,
            ), open(out_path, 'wb'))

        runtime_all += runtime
        n_processed += len(results)
        # break

    # terminates the child process
    frame_queue.put(None)

    runtime_all_np = np.array(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/opts.fps).sum()

    out_path = join(opts.out_dir, 'time_all.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump((
            runtime_all, n_processed,
            n_total, n_small_runtime,
        ), open(out_path, 'wb'))    

    # convert to ms for display
    runtime_all_np *= 1e3

    print(f'{n_processed}/{n_total} frames processed')
    print('Runtime (ms): mean: %g; std: %g; min: %g; max: %g' % (
        runtime_all_np.mean(),
        runtime_all_np.std(ddof=1),
        runtime_all_np.min(),
        runtime_all_np.max(),
    ))
    print(f'Runtime smaller than unit time interval: '
        f'{n_small_runtime}/{n_processed} '
        f'({100.0*n_small_runtime/n_processed:.4g}%)')

    print(f'Time spent on sending the frame (ms): {1e3*t_send_frame_all/n_processed}')
    print(f'Time spent on receiving the result (ms): {1e3 *t_recv_res_all/n_processed}')


if __name__ == '__main__':
    main()