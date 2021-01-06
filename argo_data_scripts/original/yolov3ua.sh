dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=yolov3ua

python det/det_coco_fmt.py \
	--data-root "$ssdDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/val.json" \
	--config "det/yolo_v3_ua/yolov3ua.py" \
	--weights "$dataDir/ModelZoo/yolo/yolov3.pt" \
	--no-mask \
	--out-dir "$dataDir/Exp/ArgoVerse1.1/output/$methodName/val" \
	--overwrite \

	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/$methodName/val" \
	# --vis-scale 0.5 \