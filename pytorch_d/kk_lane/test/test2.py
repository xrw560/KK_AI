import os
import pandas as pd
from sklearn.utils import shuffle
import shutil
import numpy as np
import cv2
from collections import  Counter
path = r"F:\data\lane\Gray_Label\Label_road02\Label\Record001\Camera 5\170927_063811892_Camera_5_bin.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(Counter(img.flatten()))
# print(img.shape)
# cv2.imshow('', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
