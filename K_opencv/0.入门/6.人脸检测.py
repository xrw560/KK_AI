from 入门.imutils import *

image = imread('Solvay.jpg')
"""
image：输入图像
scaleFactor=1.1：这个是每次缩小图像的比例，默认是1.1
minNeighbors=3：匹配成功所需要的周围矩形框的数目，每一个特征匹配到的区域都是一个矩形框，只有多个矩形框同时存在的时候，才认为是匹配成功，比如人脸，这个默认值是3。
minSize：匹配人脸的最小范围
flags=0：可以取如下这些值：
CASCADE_DO_CANNY_PRUNING=1, 利用canny边缘检测来排除一些边缘很少或者很多的图像区域
CASCADE_SCALE_IMAGE=2, 正常比例检测
CASCADE_FIND_BIGGEST_OBJECT=4, 只检测最大的物体
CASCADE_DO_ROUGH_SEARCH=8 初略的检测
"""

# 级联分类器
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(2, 2), flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
show(image)
