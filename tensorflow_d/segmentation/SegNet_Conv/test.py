# from nets.segnet import convnet_segnet
#
# model = convnet_segnet(2, input_height=416, input_width=416)
# model.summary()

from PIL import Image
import numpy as np

HEIGHT = 416
WIDTH = 416
img = Image.open(r"/home/zhouning/datasets/bili_seg/jpg" + '/' + "100.jpg")
img = img.resize((WIDTH, HEIGHT))
img = np.array(img)
img = img / 255

X_train = []
X_train.append(img)
# 从文件中读取图像
NCLASSES = 2
Y_train = []
img = Image.open(r"/home/zhouning/datasets/bili_seg/png" + '/' + "100.png")
img = img.resize((int(WIDTH / 2), int(HEIGHT / 2)))
img = np.array(img)
seg_labels = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES))
for c in range(NCLASSES):
    seg_labels[:, :, c] = (img[:, :, 0] == c).astype(int)
seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
# for index, item in enumerate(seg_labels):
#     if item[1] == 1:
#         print(index)
Y_train.append(seg_labels)
pass
