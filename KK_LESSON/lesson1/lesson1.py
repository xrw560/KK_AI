import cv2
import numpy as np


def process_an_image(img):
    """灰度化、滤波和Canny"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)
    return edges


def roi_mask(img, corner_points):
    """掩膜"""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


if __name__ == "__main__":
    width = 800  # 缩放宽度

    blur_ksize = 5  # 高斯滤波核大小
    # Canny边缘检测高低阈值
    canny_lth = 20
    canny_hth = 50
    """ 1. 读取图片"""
    image = cv2.imread('171206_064741277_Camera_5.jpg')

    """ 2. 等比例缩放"""
    high = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, high))

    """ 3. 灰度化、滤波和Canny"""
    image = process_an_image(image)

    # 掩膜区域
    area = np.array(
        [[[0, high], [width * 5 // 13, high * 6 // 13],
          [width * 7 // 13, high * 6 // 13], [width, high]]])
    """ 4. 掩膜，保留车道区域 """
    image = roi_mask(image, area)

    """ 5. 裁剪，将车辆裁剪掉"""
    image[high * 16 // 20:high, 0:width * 2 // 3] = 0
    image[high * 18 // 20:high, 0:width * 4 // 5] = 0
    image[high * 10 // 16:high, 0:width * 2 // 11] = 0

    """ 6. 显示图像"""
    cv2.imshow('gray', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """ 7. 保存图像"""
    cv2.imwrite("new_image.jpg", image)
