# Run detection in real-time setting on a COCO-format dataset
# With two seperate configurations for the detectors
raise Exception('Not updated with new detection interface')
import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter

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
from det import parse_mmdet_result, vis_det, eval_ccf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--det1-stride', type=float, default=None)
    parser.add_argument('--det1-in-scale', type=float, required=True)
    parser.add_argument('--det2-in-scale', type=float, required=True)
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

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    config = mmcv.Config.fromfile(opts.config)
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

    runtime_all = []
    which_cfg_all = []

    n_processed = 0
    n_total = 0

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
        which_cfg = []
        last_fidx = None
        
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
            frame = frames[fidx]

            if len(results) % opts.det1_stride == 0:
                cfg_id = 0
                model.cfg.data.test.img_scale = opts.det1_in_scale
            else:
                cfg_id = 1
                model.cfg.data.test.img_scale = opts.det2_in_scale

            result = inference_detector(model, frame)
            torch.cuda.synchronize()

            t2 = perf_counter()
            t_elapsed = t2 - t_start
            if t_elapsed >= t_total:
                break
            
            timestamps.append(t_elapsed)
            results.append(result)
            input_fidx.append(fidx)
            runtime.append(t2 - t1)
            which_cfg.append(cfg_id)

        out_path = join(opts.out_dir, seq + '.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump((
                results, timestamps,
                input_fidx, runtime,
            ), open(out_path, 'wb'))

        runtime_all += runtime
        which_cfg_all += which_cfg
        n_processed += len(results)

    runtime_all_np = np.array(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/opts.fps).sum()  

    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'runtime_all': runtime_all,
            'n_processed': n_processed,
            'n_total': n_total,
            'n_small_runtime': n_small_runtime,
            'which_cfg_all': which_cfg_all,
        }, open(out_path, 'wb'))  

    # For backward-compatibility
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

    which_cfg_all = np.array(which_cfg_all)
    rt_sel = runtime_all_np[which_cfg_all == 0]
    print('Det 1 Runtime (ms): mean: %g; std: %g; min: %g; max: %g' % (
        rt_sel.mean(),
        rt_sel.std(ddof=1),
        rt_sel.min(),
        rt_sel.max(),
    ))

    rt_sel = runtime_all_np[which_cfg_all == 1]
    print('Det 2 Runtime (ms): mean: %g; std: %g; min: %g; max: %g' % (
        rt_sel.mean(),
        rt_sel.std(ddof=1),
        rt_sel.min(),
        rt_sel.max(),
    ))

if __name__ == '__main__':
    main()