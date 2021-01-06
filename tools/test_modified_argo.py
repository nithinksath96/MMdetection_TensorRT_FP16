import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np
import mmcv
import torch

from pycocotools.coco import COCO
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
#sys.path.insert(0, '/home/nsathish/Efficient_object_detection/mmdetection-v100/argo_data_scripts/det')
from argo_data_scripts.util import mkdir2

from argo_data_scripts.det.det_apis import init_detector, inference_detector
from argo_data_scripts.det import imread, parse_det_result, vis_det, eval_ccf
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument(
        '--scale',
        type=float,
        help='To change the input image scale'
    )

    args = parser.parse_args()
    return args
def main():
    assert torch.cuda.device_count() == 1
   
    opts = parse_args()
    
    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    # ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign'] . Only 8. But coco has 80 classes.
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    # Mapping the above labels to coco labels: i.e person -0, bicycle - 1, car -2, etc
    coco_mapping = None if opts.no_class_mapping else db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']
    pdb.set_trace()

    # TODO_Nithin: Check if the default weights were used in the baseline
    model = init_detector(opts)
    #print(model)
    results_raw = []        # image based, all 80 COCO classes
    results_ccf = []        # instance based

    for iid, img in tqdm(db.imgs.items()):
        img_name = img['name']
        #print(iid)
        sid = img['sid']
        seq_name = seqs[sid]

        img_path = join(opts.data_root, seq_dirs[sid], img_name)
        I = imread(img_path)
        #print("Img shape", I.shape)
        '''
        TODO_Nithin: Check what is the shape the image is resized to in the pre-processing stage. 
        Currently, it is (416,416). It could be (512, 512) or (608, 608). Check if the pre-processing steps are right.
        Pre-processing steps in yolo-v3 SPP
        If augment is False
            1. Load images
            2. letter box (Resize image to a 32-pixel-multiple rectangle)
            3. Load labels and normalize labels from xywh format to xyxy format
            4. Convert image from BGR to RGB and make it as an continousarray 
            5. Convert image from uint8 to float32, 0 - 255 to 0.0 - 1.0
        if augment is True:
            Additionally you'll have 
            1. Random affine
            2. Random left-right flip
            3. Random up-down flip
            4. Augment HSV
        '''

        result = inference_detector(model, I, gpu_pre=not opts.cpu_pre)
        results_raw.append(result)
        bboxes, scores, labels, masks = \
            parse_det_result(result, coco_mapping, n_class)
        if vis_out:
            vis_path = join(opts.vis_dir, seq_name, img_name[:-3] + 'jpg')
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
    out_path = join(opts.out_dir, 'results_raw.pkl')
    
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_raw, open(out_path, 'wb'))

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

        if vis_out:
            print(f'python vis/make_videos.py "{opts.vis_dir}"')
if __name__ == '__main__':
    main()