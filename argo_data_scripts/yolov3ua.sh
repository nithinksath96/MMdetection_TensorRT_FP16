dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/ArgoVerse1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

methodName=yolov3ua

python tools/test_modified_argo.py  \
	--no-mask \
	--overwrite \
	--data-root "$ssdDir/ArgoVerse1.1/tracking" \
	--annot-path "$dataDir/ArgoVerse1.1/tracking/coco_fmt/val.json" \
	--config "argo_data_scripts/det/yolo_v3_ua/yolov3ua.py" \
	--weights "$dataDir/ModelZoo/yolo/yolov3.pt" \
	--out-dir "Exp/ArgoVerse1.1/output/$methodName/val" \
	--in-scale 1 \
	# --vis-dir "$dataDir/Exp/ArgoVerse1.1/vis/$methodName/val" \
	#--vis-scale 1 \