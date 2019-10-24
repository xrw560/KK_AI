from imutils import *
import cv2
import numpy as np

image = imread('test.jpg')
print(image.shape)

# 创建遮挡
mask = np.zeros(image.shape, dtype='uint8')
white = (255, 255, 255)
# cv2.rectangle(mask, (50, 50), (250, 350), white, -1)
cv2.circle(mask, (150, 100), 80, white, -1)
# show(mask)

# 对图像遮挡
masked = cv2.bitwise_and(image, mask)
show(masked)
