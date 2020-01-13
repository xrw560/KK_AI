import os
import pandas as pd
from sklearn.utils import shuffle
import shutil
import numpy as np
import cv2

# ================================================
# make train & validation lists
# ================================================
label_list = []
image_list = []

image_dir = 'F:\\data\\lane\\Road02\\ColorImage_road02'
label_dir = 'F:\\data\\lane\\Gray_Label'

for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')
    # print(image_sub_dir1, label_sub_dir1)

    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        # print(image_sub_dir2, label_sub_dir2)

        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            # print(image_sub_dir3, label_sub_dir3)

            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg', '_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)  # 图像文件路径
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)  # 分割label地址
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list)
print("The length of image is {} and label is {}".format(len(image_list), len(label_list)))

total_length = len(image_list)
sixth_part = int(total_length * 0.5)
eighth_part = int(total_length * 0.8)
all = pd.DataFrame({'image': image_list, 'label': label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:]

all_shuffle.to_csv('../data_list/all.csv', index=False)
train_dataset.to_csv('../data_list/train.csv', index=False)
val_dataset.to_csv('../data_list/val.csv', index=False)
test_dataset.to_csv('../data_list/test.csv', index=False)
