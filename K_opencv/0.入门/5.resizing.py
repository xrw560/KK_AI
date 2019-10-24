from 入门.imutils import *

# image = imread('image.jpg')
# print(image.shape)
# show(image)
# width = 150
# high = 150
# image = cv2.resize(image, (width, high))
# # show(image)
# print(image.shape)

image = imread('image.jpg')
width = 80
high = int(image.shape[0] * width / image.shape[1])
image = cv2.resize(image, (width, high))
# show(image)
print(image.shape)


### 5中插值方法：
# 双线性 cv2.INTER_LINEAR
image = cv2.resize(image, (width, high), interpolation=cv2.INTER_LINEAR)
# 基于像素区域 cv2.INTER_AREA
image = cv2.resize(image, (width, high), interpolation=cv2.INTER_AREA)
# 立方插值 cv2.INTER_CUBIC
image = cv2.resize(image, (width, high), interpolation=cv2.INTER_CUBIC)
# 兰索斯插值 cv2.INTER_LANCZOS4
image = cv2.resize(image, (width, high), interpolation=cv2.INTER_LANCZOS4)