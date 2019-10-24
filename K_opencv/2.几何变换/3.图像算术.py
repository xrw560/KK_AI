from imutils import *
import cv2
import numpy as np

image = imread("test.jpg")
# 图像加法
print(cv2.add(np.uint8([200]), np.uint8([100])))
# 普通加法
print(np.uint8([200]) + np.uint8([100]))  # 255+1=0

# 图像减法
print(cv2.subtract(np.uint8([50]), np.uint8([100])))
# 普通减法
print(np.uint8([50]) - np.uint8([100]))
# 生成跟图片形状相同的并且全为100的数据
M = np.ones(image.shape, dtype='uint8') * 100
# 所有的像素加100
# image = cv2.add(image, M)
# 所有的像素减100
image = cv2.subtract(image, M)
show(image)
