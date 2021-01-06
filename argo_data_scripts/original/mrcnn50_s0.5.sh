dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=mrcnn50

python det/det_coco_fmt.py \
	--eval-mask \
	--overwrite \
	--data-root "$ssdDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/htc_dconv2_ms_val.json" \
	--config "$HOME/repo/mmdetection/configs/mask_rcnn_r50_fpn_1x.py" \
	--weights "$ssdDir/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth" \
	--in-scale 0.5 \
	--out-dir "$dataDir/Exp/ArgoVerse1.1/output/mrcnn50_s0.5/val" \
	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/mrcnn50_s0.5/val" \