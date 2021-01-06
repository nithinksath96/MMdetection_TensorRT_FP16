import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

import pdb
import pycuda.driver as cuda
# import tensorrt as trt
import os
import numpy as np
# import pycuda.autoinit
import torch
import pickle
#from mmdet.models.detectors import load_trt
import sys
sys.path.append('/home/nsathish/torch2trt/')

from torch2trt import torch2trt
from time import perf_counter


torch2trt_init=False
model_trt_backbone=None
model_trt_head=None
time_torch2trt=0

model_trt_neck = None
is_fp16 = True

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def do_inference_v2(self,context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize() # Ensures all the computations are computed before the results are returned
        # Return only the host outputs.
        return [out.host for out in outputs]

    def TensorRT(self, img):
        img = img.cpu().numpy()
        img = np.asarray(img)
       
        #pdb.set_trace()

        output_shapes1 = [(1, 324, 64, 64), (1, 486, 32, 32), (1, 486, 16, 16),(1, 486, 8, 8), (1, 486, 4, 4),(1, 324, 2, 2),(1, 324, 1, 1) ]
        output_shapes2 = [(1, 16, 64, 64), (1, 24, 32, 32), (1, 24, 16, 16),(1, 24, 8, 8), (1, 24, 4, 4),(1, 16, 2, 2),(1, 16, 1, 1) ]

        # print("inputs",load_trt.inputs)
        # print("outputs",load_trt.outputs)
        # print("bindings",load_trt.bindings)
        # print("stream",load_trt.stream)
        # print("context",load_trt.context)

        load_trt.inputs[0].host = img
        trt_outputs = self.do_inference_v2(load_trt.context, bindings=load_trt.bindings, inputs=load_trt.inputs, outputs=load_trt.outputs, stream=load_trt.stream)

        res1 = [torch.from_numpy(output.reshape(shape)).float().cuda() for output, shape in zip(trt_outputs[::2], output_shapes1)]
        res2 = [torch.from_numpy(output.reshape(shape)).float().cuda() for output, shape in zip(trt_outputs[1::2], output_shapes2)]

        res =  (res1, res2)
        # for i in range(len(res)):
        #     for j in range(len(res[i])):
        #         res[i][j] = torch.from_numpy(res[i][j]).float().cuda()


        return res
    def simple_test(self, img, img_meta, rescale=False):
        global torch2trt_init
        global model_trt_backbone
        global model_trt_head
        global time_torch2trt
        global model_trt_neck
        global is_fp16

        # pdb.set_trace()

        # x = self.extract_feat(img)
        # outs = self.bbox_head(x)

        if(torch2trt_init==False):
            print("\n Using torch2trt \n")
            start = perf_counter()
            extract_feat_output_backbone = self.backbone(img)
            if(self.with_neck):
                extract_feat_output_neck=self.neck(extract_feat_output_backbone)


            model_trt_backbone = torch2trt(self.backbone, [img],fp16_mode=is_fp16)
            if(self.with_neck):
                model_trt_neck = torch2trt(self.neck, [extract_feat_output_backbone],fp16_mode=is_fp16)
                model_trt_head = torch2trt(self.bbox_head,[extract_feat_output_neck],box_class_split=True,fp16_mode=is_fp16)
 
            else:
                model_trt_head = torch2trt(self.bbox_head,[extract_feat_output_backbone],box_class_split=True,fp16_mode=is_fp16)

            end = perf_counter()
            
            time_torch2trt = (end - start)
            print("\n Time taken to initialize:", time_torch2trt) 

            torch2trt_init=True
        
        # # #Using torch2trt
        x = model_trt_backbone(img)
        if(self.with_neck):
            x = model_trt_neck(x)
        outs = model_trt_head(x)

        #Using TensorRT
        #outs = self.TensorRT(img)
        #pdb.set_trace()

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
