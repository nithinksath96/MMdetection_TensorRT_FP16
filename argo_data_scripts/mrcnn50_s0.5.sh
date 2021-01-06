dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=mrcnn50
python tools/test_modified_argo.py  \
	--no-mask \
	--overwrite \
	--data-root "$dataDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/val.json" \
	--config "/home/nsathish/Efficient_object_detection/mmdetection/configs/mask_rcnn_r50_fpn_1x.py" \
	--weights "$ssdDir/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth" \
	--in-scale 1 \
	--out-dir "Exp/ArgoVerse1.1/output/mrcnn50_s0.5/val" \
	--scale=1
	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/mrcnn50_s0.5/val" \