dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=frcnn50
python tools/test_modified_argo.py  \
	--no-mask \
	--overwrite \
	--data-root "$dataDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/val.json" \
	--config "/data2/nsathish/mmdetection-v100/configs/fast_rcnn_r50_fpn_1x_scale_modified.py" \
	--weights "/data2/nsathish/results/work_dirs/faster_rcnn_r50_fpn_1x_iou_modified/epoch_11.pth" \
	--in-scale 1 \
	--out-dir "/data2/nsathish/results/ArgoVerse1.1/output/frcnn50_s0.5/val" \
	#--scale=1
	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/mrcnn50_s0.5/val" \