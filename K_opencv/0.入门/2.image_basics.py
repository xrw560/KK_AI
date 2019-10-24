import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")
(h, w, c) = image.shape
print(image.shape)

(b, g, r) = image[0, 0]
print(image[0, 0])

image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print(image[0, 0])

cX, cY = (w // 2, h // 2)
tl = image[0:cY, 0:cX]  # top left
tr = image[0:cY, cX:w]
bl = image[cY:h, 0:cX]
br = image[cY:h, cX:w]


def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# show(tl)
# show(tr)
# show(bl)
# show(br)

image[0:cY, 0:cX] = (0, 0, 255)
show(image)
