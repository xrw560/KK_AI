import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)

for item in dataset:
    print(item)

