from os.path import dirname
from types import MethodType
import numpy as np
from PIL import Image
import pickle, json, cv2, mmcv

import pycocotools.mask as maskUtils
from pycocotools.cocoeval import COCOeval

#from util import mkdir2
from argo_data_scripts.util import mkdir2

def imread(path, method='PIL'):
    if method == 'PIL':
        return np.asarray(Image.open(path))
    else:
        return mmcv.imread(path)


def imwrite(img, path, method='PIL', auto_mkdir=True):
    if method == 'PIL':
        if auto_mkdir:
            mkdir2(dirname(path))
        Image.fromarray(img).save(path)
    else:
        mmcv.imwrite(img, path, auto_mkdir=auto_mkdir)

def parse_det_result(result, class_mapping=None, n_class=None, separate_scores=True, return_sel=False):
    if len(result) > 2:
        bboxes_scores, labels, masks = result
    else:
        bboxes_scores, labels = result
        masks = None

    if class_mapping is not None:
        labels = class_mapping[labels]
        sel = labels < n_class
        bboxes_scores = bboxes_scores[sel]
        labels = labels[sel]
        if masks is not None:
            masks = masks[sel]
    else:
        sel = None
    if separate_scores:
        if len(labels):
            bboxes = bboxes_scores[:, :4]
            scores = bboxes_scores[:, 4]
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        outs = [bboxes, scores, labels, masks]
    else:
        outs = [bboxes_scores, labels, masks]
    if return_sel:
        outs.append(sel)
    return tuple(outs)

def decode_mask(mask_encoded, sel=None):
    if mask_encoded is None:
        return np.array([])
    masks = mask_encoded()
    masks = np.asarray(masks)
    if sel is not None:
        masks = masks[sel]
    return masks

def parse_mmdet_result(result, class_mapping=None, n_class_mapped=None, class_subset=None):
    if isinstance(result, tuple):
        bbox_result, mask_result = result
    else:
        bbox_result, mask_result = result, None
    
    if class_mapping is not None:
        mapped = [np.empty((0, 5)) for c in range(n_class_mapped)]
        for c, row in enumerate(bbox_result):
            if c in class_mapping:
                mapped[class_mapping[c]] = row
        bbox_result = mapped

        if mask_result is not None:
            mapped = [np.empty((0, 5)) for c in range(n_class_mapped)]
            for c, row in enumerate(mask_result):
                if c in class_mapping:
                    mapped[class_mapping[c]] = row
            mask_result = mapped
    elif class_subset is not None:
        bbox_result = [bbox_result[c] for c in class_subset]
        if mask_result is not None:
            mask_result = [mask_result[c] for c in class_subset]

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    scores = bboxes[:, -1]
    bboxes = bboxes[:, :4]

    if mask_result is None:
        masks = None
    else:
        masks = mmcv.concat_list(mask_result)

    # bboxes in the form of n*[left, top, right, bottom]
    return bboxes, scores, labels, masks


def vis_det(img, bboxes, labels, class_names,
    masks=None, scores=None, score_th=0,
    out_scale=1, out_file=None):
    # img with RGB channel order
    # bboxes in the form of n*[left, top, right, bottom]

    if out_scale != 1:
        img = mmcv.imrescale(img, out_scale, interpolation='bilinear')

    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    if masks is not None:
        masks = np.asarray(masks)

    empty = len(bboxes) == 0
    if not empty and scores is not None and score_th > 0:
        sel = scores >= score_th
        bboxes = bboxes[sel]
        labels = labels[sel]
        scores = scores[sel]
        if masks is not None:
            masks = masks[sel]
        empty = len(bboxes) == 0

    if empty:
        if out_file is not None:
            imwrite(img, out_file)
        return img

    if out_scale != 1:
        bboxes = out_scale*bboxes
        # we don't want in-place operations like bboxes *= out_scale

    if masks is not None:
        img = np.array(img) # make it writable
        for mask in masks:
            color = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8
            )
            # color = np.array((138, 214, 242), np.uint8)
            m = maskUtils.decode(mask)
            if out_scale != 1:
                m = mmcv.imrescale(
                    m.astype(np.uint8), out_scale,
                    interpolation='nearest'
                )
            m = m.astype(np.bool)
            img[m] = 0.5*img[m] + 0.5*color

    bbox_color = (0, 255, 0)
    text_color = (0, 255, 0)
    thickness = 1
    font_scale = 0.5

    bboxes = bboxes.round().astype(np.int32)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        lt = (bbox[0], bbox[1])
        rb = (bbox[2], bbox[3])
        cv2.rectangle(
            img, lt, rb, bbox_color, thickness=thickness
        )
        if class_names is None:
            label_text = f'class {label}'
        else:
            label_text = class_names[label]
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        cv2.putText(
            img, label_text, (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX, font_scale,
            text_color,
        )

    if out_file is not None:
        imwrite(img, out_file)
    return img


def eval_ccf(db, results, class_subset=None, per_class=False, iou_type='bbox'):
    # ccf means CoCo Format
    if isinstance(results, str):
        if results.endswith('.pkl'):
            results = pickle.load(open(results, 'rb'))
        else:
            results = json.load(open(results, 'r'))
    # if iou_type == 'segm':
    #     for i in range(len(results)):
    #         results[i]['segmentation']['counts'] = results[i]['segmentation']['counts'].decode()
    #         # det.pop('bbox', None)
    results = db.loadRes(results)
    cocoEval = COCOeval(db, results, iou_type)
    if class_subset is not None:
        cocoEval.params.catIds = class_subset
        

    def summarize_per_class(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                # print(s.shape)
                for k in range(s.shape[2]):
                    print(np.mean(s[:, :, k]))
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            raise NotImplementedError
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    
    # r = 8.417032065165149 # ArgoVerse (1920x1200) / mean COCO
    # for rng in cocoEval.params.areaRng:
    #     rng[0] *= r
    #     rng[1] *= r

    cocoEval.evaluate()
    cocoEval.accumulate()
    if per_class:
        cocoEval.summarize_per_class = MethodType(summarize_per_class, cocoEval)
        cocoEval.summarize_per_class()
    else:
        cocoEval.summarize()

    return {
        'eval': cocoEval.eval,
        'stats': cocoEval.stats,
    }

def result_from_ccf(ccf, iid, start_idx=None, mask=True, sequential=True):
    ''' Get the detections of particular image id '''
    if sequential:
        while start_idx < len(ccf) and ccf[start_idx]['image_id'] < iid:
            start_idx += 1
        end_idx = start_idx
        while end_idx < len(ccf) and ccf[end_idx]['image_id'] == iid:
            end_idx += 1
        dets = ccf[start_idx:end_idx]
    else:
        dets = [r for r in ccf if r['image_id'] == iid]
    
    bboxes = np.array([d['bbox'] for d in dets])
    scores = np.array([d['score'] for d in dets])
    labels = np.array([d['category_id'] for d in dets])
    if mask:
        if len(dets) and 'segmentation' in dets[0]:
            masks = np.array([d['segmentation'] for d in dets])
        else:
            masks = None
        return end_idx, bboxes, scores, labels, masks
    else:
        return end_idx, bboxes, scores, labels
