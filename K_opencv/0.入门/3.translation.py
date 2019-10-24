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
M = np.float32([[1, 0, 250], [0, 1, 500]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# show(shifted)

image = imread('image.jpg')
M = np.float32([[1, 0, -250], [0, 1, -500]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
show(shifted)
