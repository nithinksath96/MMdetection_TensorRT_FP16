from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# # build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
# # test a single image and show the results

#img = '/home/nsathish/Efficient_object_detection/mmdetection/demo.jpg'   or
img = mmcv.imread('/home/nsathish/Efficient_object_detection/mmdetection/demo.jpg')
#print("Image",img)
#print("ModeL",model)
result = inference_detector(model, img)
# visualize the results in a new window
#print(result)
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='/home/nsathish/Efficient_object_detection/mmdetection/result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     show_result(frame, result, model.CLASSES, wait_time=1)
