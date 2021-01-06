import cv2 as cv
import os

data_path = "/data/mengtial/COCO/val2017/"
images = []
max_h=float('-inf')
max_w=float('-inf')
for filename in os.listdir(data_path):
    img = cv.imread(os.path.join(data_path,filename))
    max_h = max(max_h,img.shape[0])
    max_w = max(max_w,img.shape[1])
    # print(img.shape[0])
    # print(img.shape[1])
print(max_h)
print(max_w)
