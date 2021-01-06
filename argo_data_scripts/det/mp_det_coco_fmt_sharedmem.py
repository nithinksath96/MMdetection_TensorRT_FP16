# Run detection in real-time setting on a COCO-format dataset
import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import sharedmem
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

img_width, img_height = 1920, 1200

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

def det_process(opts, in_pipe, out_pipe, frame_buffer):
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
        _ = inference_detector(model, np.zeros((img_height, img_width, 3), np.uint8))
        torch.cuda.synchronize()

        # get images ready
        frames = frame_buffer
        # frames = np.frombuffer(, np.uint8)
        # frames = frames.reshape(-1, img_height, img_width, 3)
        # note this buffer will be updated by the main process without notice
        # but for the provided fidx, the image is always up-to-date

        # signal ready, no errors
        out_pipe.send(None)

        # import cProfile
        while 1: # entering daemon mode
            fidx = in_pipe.recv()
            if fidx is None:
                # exit flag
                break

            
            fidx, t1 = fidx
            t4 = perf_counter() 
            frame = frames[fidx].copy()

            t2 = perf_counter() 
            t_send_frame = t2 - t1
            t_copy = t2 - t4

            # pr = cProfile.Profile()
            # pr.enable()

            result = inference_detector(model, frame)
            torch.cuda.synchronize()
            # pr.disable()

            t3 = perf_counter()
            print('send: %.3g det: %.3g' % (t_send_frame, t3 - t2))

            # res_buffer = result
            out_pipe.send([result, t_send_frame, t3, t_copy])

        # pr.dump_stats('mrcnn50_nm_s0.5_1080ti_with_shm_input.prof')

    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        out_pipe.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()
    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    # CUDA runtime does not support the fork
    mp.set_start_method('spawn')

    # calculate the image size for shared memory allocation
    if 'max_seq_len' in db.dataset:
        max_seq_len = db.dataset['max_seq_len']
    else:
        seq_len = [len([img for img in db.imgs.values() if img['sid'] == sid])
            for sid in range(len(seqs))]
        max_seq_len = max(seq_len)
    frame_buffer = sharedmem.empty((max_seq_len, img_height, img_width, 3), 'uint8')
    frames = frame_buffer
    child_recv, parent_send = mp.Pipe(False)
    parent_recv, child_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, child_recv, child_send, frame_buffer))
    det_proc.start()


    runtime_all = []
    n_processed = 0
    n_total = 0

    t_send_frame_all = 0
    t_recv_res_all = 0

    init_error = parent_recv.recv() # wait till the detector is ready
    if isinstance(init_error, Exception):
        raise init_error

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        # load all frames in advance
        for i, img in enumerate(frame_list):
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            frames[i] = mmcv.imread(img_path)
        n_frame = len(frame_list)
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

            t_start_frame = perf_counter()
            parent_send.send((fidx, t_start_frame))
            result = parent_recv.recv() # wait
            if isinstance(result, Exception):
                raise result
            result, t_send_frame, t_start_res = result

            t2 = perf_counter()

            t_send_frame_all += t_send_frame
            t_recv_res_all += t2 - t_start_res
            print('recv: %.3g elapsed: %.3g' % (t2 - t_start_res, t2 - t1))
            
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
    parent_send.send(None)

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