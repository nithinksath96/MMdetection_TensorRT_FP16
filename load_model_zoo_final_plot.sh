#!/bin/bash
# For FP32 

python tools/test_modified_m.py configs/mask_rcnn_r50_fpn_1x.py checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth  --eval bbox --scale=1.0
python tools/test_modified_m.py configs/mask_rcnn_r50_fpn_1x.py checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth  --eval bbox --scale=0.75
python tools/test_modified_m.py configs/mask_rcnn_r50_fpn_1x.py checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth  --eval bbox --scale=0.5
python tools/test_modified_m.py configs/mask_rcnn_r50_fpn_1x.py checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth  --eval bbox --scale=0.25


python tools/test_modified_m.py configs/mask_rcnn_r101_fpn_1x.py  checkpoints/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth  --eval bbox  --scale=1.0
python tools/test_modified_m.py configs/mask_rcnn_r101_fpn_1x.py  checkpoints/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth  --eval bbox  --scale=0.75
python tools/test_modified_m.py configs/mask_rcnn_r101_fpn_1x.py  checkpoints/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth  --eval bbox  --scale=0.5
python tools/test_modified_m.py configs/mask_rcnn_r101_fpn_1x.py  checkpoints/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth  --eval bbox  --scale=0.25


python tools/test_modified.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth --eval bbox --scale=1.0
python tools/test_modified.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth --eval bbox --scale=0.75
python tools/test_modified.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth --eval bbox --scale=0.5
python tools/test_modified.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth --eval bbox --scale=0.25

python tools/test_modified.py configs/retinanet_r101_fpn_1x.py checkpoints/retinanet_r101_fpn_1x_20181129-f016f384.pth --eval bbox --scale=1.0
python tools/test_modified.py configs/retinanet_r101_fpn_1x.py checkpoints/retinanet_r101_fpn_1x_20181129-f016f384.pth --eval bbox --scale=0.75
python tools/test_modified.py configs/retinanet_r101_fpn_1x.py checkpoints/retinanet_r101_fpn_1x_20181129-f016f384.pth --eval bbox --scale=0.5
python tools/test_modified.py configs/retinanet_r101_fpn_1x.py checkpoints/retinanet_r101_fpn_1x_20181129-f016f384.pth --eval bbox --scale=0.25

python tools/test_modified.py configs/ssd512_coco.py  checkpoints/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth --eval bbox --scale=1.0
python tools/test_modified.py configs/ssd512_coco.py  checkpoints/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth --eval bbox --scale=0.75
python tools/test_modified.py configs/ssd512_coco.py  checkpoints/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth --eval bbox --scale=0.5
python tools/test_modified.py configs/ssd512_coco.py  checkpoints/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth --eval bbox --scale=0.25

