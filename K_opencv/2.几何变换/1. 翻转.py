from imutils import *
import cv2

image = imread("test.jpg")
show(image)
image = cv2.flip(image, 1)  # 1水平翻转，0垂直翻转，-1:水平+垂直
show(image)
