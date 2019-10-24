import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")
# print("width:%d pixels" % (image.shape[1]))
# print("height: %d pixels" % (image.shape[0]))
# print("channels: %d pixels" % (image.shape[2]))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.axis('off')
# plt.show()

cv2.imwrite("new_image.jpg", image)
