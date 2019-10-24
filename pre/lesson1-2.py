import cv2

img = cv2.imread("F:/data/zhaoliying.jpeg", 1)  # 0读灰度图像,1:原图
img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
print(img.shape)
img = img[30:270, 100:400]
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -40, 1)
img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("zhaoliying", img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
