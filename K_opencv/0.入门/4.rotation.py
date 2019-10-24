import cv2
import matplotlib.pyplot as plt
import numpy as np


def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def imread(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


image = imread('image.jpg')
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)
# (cX,cY)-旋转中心点
# 45-逆时针旋转45度
# 1.0-缩放
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
image = cv2.warpAffine(image, M, (w, h))
# show(image)

