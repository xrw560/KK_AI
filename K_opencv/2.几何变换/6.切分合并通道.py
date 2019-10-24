from imutils import *
import cv2
import numpy as np

image = imread('test.jpg')
(R, G, B) = cv2.split(image)
print(R.shape, G.shape, B.shape)
merged = cv2.merge([R, G, B])
# show(merged)

cv2.imshow('R', R)
cv2.imshow('G', G)
cv2.imshow('B', B)
cv2.waitKey(0)
cv2.destroyAllWindows()
