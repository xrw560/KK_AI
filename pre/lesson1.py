import cv2
import random
import numpy as np

img_gray = cv2.imread("F:/data/lenna.jpg", 0)  # 0读灰度图像,1:原图
img = cv2.imread("F:/data/lenna.jpg", 1)  # 0读灰度图像,1:原图

# cv2.imshow("lenna", img_gray)
# key = cv2.waitKey()
# if key == 27:  # 27:ESC
#     cv2.destroyAllWindows()

print(img_gray.shape)
print(img.shape)

# image crop
img_crop = img[0:100, 0:200]


# cv2.imshow("img_crop", img_crop)

# color split
# B, G, R = cv2.split(img)
# cv2.imshow('B', B)
# cv2.imshow('G', G)
# cv2.imshow('R', R)

def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - b_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - b_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    return img_merge


img_random_color = random_light_color(img)
cv2.imshow("img_random_color", img_random_color)
key = cv2.waitKey()
if key == 27:  # 27:ESC
    cv2.destroyAllWindows()
