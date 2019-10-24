from imutils import *
import cv2
import numpy as np

# image = imread('image.jpg')
# print(image.shape)
# # for i in range(4):
# #     image = cv2.pyrDown(image)
# #     print(image.shape)
#
# for i in range(4):
#     image = cv2.pyrUp(image)
#     print(image.shape)


# 拉普拉斯金字塔
image = imread('image.jpg')
down_image1 = cv2.pyrDown(image)
down_image2 = cv2.pyrDown(down_image1)
up_image = cv2.pyrUp(down_image2)
laplacian = down_image1 - up_image
show(laplacian)
