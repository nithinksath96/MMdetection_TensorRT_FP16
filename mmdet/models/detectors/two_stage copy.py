import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin

import sys
sys.path.append('/home/nsathish/torch2trt/')

from torch2trt import torch2trt
from time import perf_counter

torch2trt_init=False
model_trt_backbone=None
model_trt_head=None
model_trt_neck=None
model_trt_test_bbox = None
is_fp16=False
time_torch2trt=0
import pdb

@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta,
                                                      self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        #pdb.set_trace()
        global torch2trt_init
        global model_trt_backbone
        global model_trt_head
        global model_trt_neck
        global model_trt_test_bbox
        global time_torch2trt
        global is_fp16

        #pdb.set_trace()
        # x = self.extract_feat(img)

        # if proposals is None:
        #     proposal_list = self.simple_test_rpn(x, img_meta,
        #                                          self.test_cfg.rpn)
        # else:
        #     proposal_list = proposals

        if(torch2trt_init==False):
            print("\n Using torch2trt \n")
            start = perf_counter()

            extract_feat_output_backbone = self.backbone(img)
            if(self.with_neck):
                extract_feat_output_neck=self.neck(extract_feat_output_backbone)
                proposal_list_trt = self.simple_test_rpn(extract_feat_output_neck, img_meta,
                                                    self.test_cfg.rpn)
                rois_trt = bbox2roi(proposal_list_trt)
                roi_feats_trt_neck = self.bbox_roi_extractor(
                                extract_feat_output_neck[:len(self.bbox_roi_extractor.featmap_strides)], rois_trt)
            else:
                proposal_list_trt = self.simple_test_rpn(extract_feat_output_backbone, img_meta,
                                                    self.test_cfg.rpn)
                rois_trt = bbox2roi(proposal_list_trt)
                roi_feats_trt_without_neck = self.bbox_roi_extractor(
                                extract_feat_output_backbone[:len(self.bbox_roi_extractor.featmap_strides)], rois_trt)


            
            model_trt_backbone = torch2trt(self.backbone, [img],fp16_mode=is_fp16)
            if(self.with_neck):
                model_trt_neck = torch2trt(self.neck, [extract_feat_output_backbone],fp16_mode=is_fp16)
                model_trt_head = torch2trt(self.rpn_head,[extract_feat_output_neck],box_class_split=True,fp16_mode=is_fp16)
                #pdb.set_trace()
                model_trt_test_bbox = torch2trt(self.bbox_head,[roi_feats_trt_neck],box_class_split=True , fp16_mode=is_fp16, max_batch_size=1000)
            else:
                model_trt_head = torch2trt(self.rpn_head,[extract_feat_output_backbone],box_class_split=True,fp16_mode=is_fp16)
                model_trt_test_bbox = torch2trt(self.bbox_head,[roi_feats_trt_without_neck],box_class_split=True , fp16_mode=is_fp16, max_batch_size=1000)
            


            end = perf_counter()
            
            time_torch2trt = (end - start)
            print("\n Time taken to initialize:", time_torch2trt)

            # import os
            # if os.path.exists("/home/nsathish/Efficient_object_detection/mmdetection/mmdet/models/detectors/det_labels_trt.txt"):
            #     os.remove("/home/nsathish/Efficient_object_detection/mmdetection/mmdet/models/detectors/det_labels_trt.txt")
            #     os.remove("/home/nsathish/Efficient_object_detection/mmdetection/mmdet/models/detectors/det_bboxes_trt.txt")
            #     print("Deleted")
            # else:
            #     print("The file does not exist") 

            torch2trt_init=True

        

        
        # #Torch2trt starts from here
        x = model_trt_backbone(img)
        if(self.with_neck):
            x = model_trt_neck(x)
        if(proposals is None):
            rpn_head_trt = model_trt_head(x)
            proposal_inputs = rpn_head_trt + (img_meta,self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals


        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
    
        # #Have to see why there is a change in values?
        #cls_score, bbox_pred = self.bbox_head(roi_feats)

        
        cls_score, bbox_pred = model_trt_test_bbox(roi_feats)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        # #pdb.set_trace()
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score[0],
            bbox_pred[0],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        

        



        #pdb.set_trace()
        #pdb.set_trace()
        # det_bboxes, det_labels = self.simple_test_bboxes(
        #     x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        # with open("/home/nsathish/Efficient_object_detection/mmdetection/mmdet/models/detectors/det_labels_trt_working.txt", "a") as myfile:
        #     myfile.write(str(det_labels) +"\n")
        # with open("/home/nsathish/Efficient_object_detection/mmdetection/mmdet/models/detectors/det_bboxes_partial_trt.txt", "a") as myfile:
        #     myfile.write(str(det_bboxes) + "\n")  
        #pdb.set_trace()

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
