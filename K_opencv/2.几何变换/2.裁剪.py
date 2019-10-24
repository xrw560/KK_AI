from imutils import *
import cv2

image = imread("test.jpg")
image1 = image[0:200, 50:200]
show(image1)
image2 = image[200:, 50:-50]
show(image)
