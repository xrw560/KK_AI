import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# root_dir = "F:\\data\\lane\\Labels_Fixed\\Label_road02\\Label"
# all_image_label = []
# for s1 in os.listdir(root_dir):
#
#     s11 = os.path.join(root_dir, s1)
#     # print(dir)
#     for s2 in os.listdir(s11):
#         s22 = os.path.join(s11, s2)
#         for s3 in os.listdir(s22):
#             s33 = os.path.join(s22, s3)
#             all_image_label.append(s33)
#
# t1 = cv2.imread(all_image_label[0])
# shape = t1.shape
# h = 600
# w = int(shape[0] * h / shape[1])
import pandas as pd
data = pd.read_csv(os.path.join(os.getcwd(), "data_list", "all.csv"),
                                header=1,
                                names=["image", "label"])

heatmap = np.zeros(cv2.imread(data['label'][0]).shape)
for item in data['label']:
    item_label = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
    heatmap += item_label
# heatmap = cv2.resize(heatmap, (h, w), interpolation=cv2.INTER_CUBIC)
print(heatmap.shape)
# t = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# plt.imshow(heatmap)
# plt.show()
# cv2.imshow('result.jpg', heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

total_sum_val = np.sum(heatmap)
total_val = np.zeros(heatmap.shape[0])
for i in range(heatmap.shape[0]):
    total_val[i] = np.sum(heatmap[i, :]) / total_sum_val

# reversed_total_val = total_val[::-1]
li = np.cumsum(total_val)
for (index, item) in enumerate(li):
    print(index, item)

# cv2.line(heatmap, (0, 788), (3384, 788), (255, 0, 0), 5)
# plt.imshow(heatmap)
# plt.show()
