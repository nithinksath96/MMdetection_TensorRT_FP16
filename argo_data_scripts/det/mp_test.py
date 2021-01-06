# Run detection in real-time setting on a COCO-format dataset

import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter, sleep
from multiprocessing import Process, Pipe

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
    parser.add_argument('--data-root', type=str, default='D:/Data/ArgoVerse/tracking')
    parser.add_argument('--annot-path', type=str, default='D:/Data/ArgoVerse/tracking/coco_fmt/htc_dconv2_ms_val.json')
    parser.add_argument('--det-stride', type=float, default=1)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(frame_recv, det_res_send):
    while 1:
        frame = frame_recv.recv()
        if frame is None:
            break
        # print(f'det: recieved {frame}')
        sleep(0.001)
        result = frame
        det_res_send.send(result)
        # print(f'det: sent out {result}')
        # result = inference_detector(model, frame)

def main():
    opts = parse_args()

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    frame_recv, frame_send = Pipe(False)
    det_res_recv, det_res_send = Pipe(False)

    p = Process(target=det_process, args=(frame_recv, det_res_send))
    # model = init_detector(config, opts.weights)
    # model.eval()

    # warm up the GPU
    # _ = inference_detector(model, np.zeros((1200, 1920, 3), np.uint8))
    # torch.cuda.synchronize()

    runtime_all = []
    n_processed = 0
    n_total = 0

    # p.daemon=True
    p.start()
    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        # load all frames in advance
        frames = []
        for img in frame_list:
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            # frames.append(mmcv.imread(img_path))
            frames.append(img_path)
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

            frame_send.send(frame)
            # print(f'main: sent {frame}')
            while not det_res_recv.poll():
                result = det_res_recv.recv()
                # print(f'main: recieved {result}')
                break

            t2 = perf_counter()
            t_elapsed = t2 - t_start
            if t_elapsed >= t_total:
                break
            
            timestamps.append(t_elapsed)
            results.append(result)
            input_fidx.append(fidx)
            runtime.append(t2 - t1)

        runtime_all += runtime
        n_processed += len(results)
        break

    # terminates the child process
    frame_send.send(None)

    runtime_all_np = np.array(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/opts.fps).sum()

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

if __name__ == '__main__':
    main()