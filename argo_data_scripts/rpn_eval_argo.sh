dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=rpn

python tools/test_modified_argo_rpn_eval.py  \
	--no-mask \
	--overwrite \
    --rpn \
	--data-root "$ssdDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/val.json" \
	--config "/home/nsathish/Efficient_object_detection/mmdetection-v100/configs/rpn_r50_fpn_1x.py" \
	--weights "/home/nsathish/Efficient_object_detection/mmdetection-v100/checkpoints/rpn_r50_fpn_1x_20181010-4a9c0712.pth" \
	--out-dir "Exp/ArgoVerse1.1/output/$methodName/val" \
	--in-scale 1 \
	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/$methodName/val" \
	#--vis-scale 1 \