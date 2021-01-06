# Run detection in real-time setting on a COCO-format dataset

import argparse, json, pickle
from os.path import join, isfile
from time import perf_counter

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO
from mmcv.runner import load_checkpoint

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from det import imread, parse_det_result, vis_det, eval_ccf
from det.det_apis import \
    init_detector, inference_detector, \
    ImageTransform, ImageTransformGPU, _prepare_data
import train.models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--weights-base', type=str, default=None)
    parser.add_argument('--n-history', type=int, default=None)
    parser.add_argument('--n-future', type=int, default=None)    
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = None if opts.no_class_mapping else db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    model = init_detector(opts)
    if opts.weights_base is not None:
        # for distillation purpose
        load_checkpoint(model, opts.weights_base)
    if opts.cpu_pre:
        img_transform = ImageTransform(
            size_divisor=model.cfg.data.test.size_divisor, **model.cfg.img_norm_cfg)
    else:
        img_transform = ImageTransformGPU(
            size_divisor=model.cfg.data.test.size_divisor, **model.cfg.img_norm_cfg)
    device = next(model.parameters()).device  # model device
    n_history = model.cfg.data.train.n_history if opts.n_history is None else opts.n_history
    n_future = model.cfg.data.train.n_future if opts.n_future is None else opts.n_future

    results_ccf = []        # instance based
    runtime_all = []
    for sid, seq in enumerate(tqdm(seqs)):
        # print(seq)
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        n_frame = len(frame_list)

        # load all frames in advance
        frames = []
        for img in frame_list:
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            frames.append(imread(img_path))
        
        with torch.no_grad():
            preprocessed = []
            for i in range(n_history):
                data = _prepare_data(frames[i], img_transform, model.cfg, device)
                preprocessed.append(data)
            for ii in range(n_history, n_frame - n_future):
                # target frame
                iid = frame_list[ii + n_future]['id']
                img_name = frame_list[ii + n_future]['name']
                I = frames[ii + n_future]

                t_start = perf_counter()
                # input frame
                data = _prepare_data(frames[ii], img_transform, model.cfg, device)
                # if n_history == 0:
                #     data_merge = data
                #     # print(data['img'])
                #     # print(data['img'][0].shape)
                #     # print(data['img'][0][0][0][300][300:305])
                #     # import sys
                #     # sys.exit()
                # else:
                preprocessed.append(data)
                # print(preprocessed[0]['img'][0].data_ptr())
                # print(preprocessed[2]['img'][0].data_ptr())
                # print(torch.all(preprocessed[0]['img'][0] == preprocessed[2]['img'][0]))
                imgs = [d['img'][0] for d in preprocessed]
                imgs = torch.cat(imgs, 0)
                imgs = imgs.unsqueeze(0)
                data_merge = {
                    'img': [imgs],
                    'img_meta': data['img_meta'],
                }
                # print(data_merge['img'][0][0][2][0][300][300:305])
                # import sys
                # sys.exit()
                result = model(return_loss=False, rescale=True, numpy_res=True, **data_merge)
                bboxes, scores, labels, masks = \
                    parse_det_result(result, coco_mapping, n_class)
                # if ii == 2:
                #     print(ii, scores)
                #     import sys
                #     sys.exit()

                # if n_history != 0:
                del preprocessed[0]
                t_end = perf_counter()
                runtime_all.append(t_end - t_start)

                if vis_out:
                    vis_path = join(opts.vis_dir, seq, img_name[:-3] + 'jpg')
                    if opts.overwrite or not isfile(vis_path):
                        vis_det(
                            I, bboxes, labels,
                            class_names, masks, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path
                        )

                # convert to coco fmt
                n = len(bboxes)
                if n:
                    bboxes[:, 2:] -= bboxes[:, :2]

                for i in range(n):
                    result_dict = {
                        'image_id': iid,
                        'bbox': bboxes[i],
                        'score': scores[i],
                        'category_id': labels[i],
                    }
                    if masks is not None:
                        result_dict['segmentation'] = masks[i]
                    results_ccf.append(result_dict)


    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'runtime_all': runtime_all,
            'n_total': len(runtime_all),
        }, open(out_path, 'wb'))  

    # convert to ms for display
    s2ms = lambda x: 1e3*x

    print_stats(runtime_all, 'Runtime (ms)', cvt=s2ms)

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}"')

if __name__ == '__main__':
    main()