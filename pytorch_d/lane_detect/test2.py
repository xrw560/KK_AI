import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


def crop_resize_data(image, label=None, image_size=(1024, 384), offset=886):
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    roi_image = image[offset:, :]
    if label is not None:
        roi_label = label[offset:, :]
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image


data = pd.read_csv(os.path.join(os.getcwd(), "data_list", "train.csv"), header=0,
                   names=["image", "label"])
images = data["image"].values[1:]
labels = data["label"].values[1:]

ori_image = cv2.imread(images[0])
ori_mask = cv2.imread(labels[0], cv2.IMREAD_GRAYSCALE)
train_img, train_mask = crop_resize_data(ori_image, ori_mask)
print(train_img.shape)
print(train_mask.shape)
