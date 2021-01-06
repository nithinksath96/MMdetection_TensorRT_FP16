# Run detection in real-time setting on a COCO-format dataset
# profiling version

import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter

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
    # parser.add_argument('--data-root', type=str, required=True)
    # # parser.add_argument('--annot-path', type=str, required=True)
    # parser.add_argument('--annot-path', type=str, default='D:/Data/ArgoVerse/tracking/coco_fmt/htc_dconv2_ms_val.json')
    # parser.add_argument('--split', type=str, default='val')
    # # parser.add_argument('--in-scale', type=float, default=None)
    # parser.add_argument('--in-scale', type=float, default=0.5)
    # parser.add_argument('--fps', type=float, default=30)
    # # parser.add_argument('--no-mask', action='store_true', default=False)
    # parser.add_argument('--no-mask', action='store_true', default=True)
    # # parser.add_argument('--out-dir', type=str, required=True)
    # parser.add_argument('--out-dir', type=str, default='D:/Data/Exp/ArgoVerse-debug/output/rt_mask_rcnn_r50/s0.5_val')
    # # parser.add_argument('--config', type=str, default='../mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    # # parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    # parser.add_argument('--config', type=str, default='../mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    
    # # parser.add_argument('--overwrite', action='store_true', default=False)
    # parser.add_argument('--overwrite', action='store_true', default=True)


    parser.add_argument('--data-root', type=str, default='D:/Data/ArgoVerse/tracking')
    parser.add_argument('--annot-path', type=str, default='D:/Data/ArgoVerse/tracking/coco_fmt/htc_dconv2_ms_val.json')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--in-scale', type=float, default=1)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, default='D:/Data/Exp/ArgoVerse-debug/output/rt_mask_rcnn_r50/s0.5_val')
    # parser.add_argument('--config', type=str, default='~/repo/mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='/scratch/mengtial/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    # parser.add_argument('--config', type=str, default='D:/Repo/mmdetection/configs/mask_rcnn_r50_fpn_1x.py')
    # parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth')
    parser.add_argument('--config', type=str, default='D:/Repo/mmdetection/configs/mask_rcnn_r101_fpn_1x.py')
    parser.add_argument('--weights', type=str, default='D:/Data/ModelZoo/mmdet/mask_rcnn_r101_fpn_2x_20181129-a254bdfc.pth')


 
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts



def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    model = init_detector(opts)

    # warm up the GPU
    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']
    _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
    torch.cuda.synchronize()

    runtime_all = []
    n_processed = 0
    n_total = 0

    # global pr
    import cProfile
    pr = cProfile.Profile()
    

    for sid, seq in enumerate(tqdm(seqs)):
        # if sid > 1:
        #     break
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        # load all frames in advance

        frames = []
        for img in frame_list:
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            frames.append(imread(img_path))
        n_frame = len(frames)
        n_total += n_frame
        
        timestamps = []
        results = []
        input_fidx = []
        runtime = []
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

            pr.enable()
            result = inference_detector(model, frame)
            torch.cuda.synchronize()
            pr.disable()

            t2 = perf_counter()
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


    pr.dump_stats('_par_/mrcnn50_s0.5_1080ti_gpupre_blocking.prof')
    # pr.dump_stats('debug.prof')

    runtime_all_np = np.array(runtime_all)
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