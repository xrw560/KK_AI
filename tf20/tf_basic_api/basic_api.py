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

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])

# index索引
# print(t)
# print(t[:, 1:])
# print(t[..., 1])

# op
# print(t + 10)
# print(tf.square((t)))
# print(t @ tf.transpose(t))

# numpy conversion
# print(t.numpy())
# print(np.square(t))  # 可以作为numpy的输入
# np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(tf.constant(np_t))

# Scalars 数值
# t = tf.constant(2.718)
# print(t.numpy())
# print(t.shape)

# strings，字符串
# t = tf.constant("cafe")
# print(t)
# print(tf.strings.length(t))
# print(tf.strings.length(t, unit="UTF8_CHAR"))
# print(tf.strings.unicode_decode(t, "UTF8"))

# string array
# t = tf.constant(["cafe", 'coffe', '咖啡'])
# print(tf.strings.length(t, unit="UTF8_CHAR"))
# r = tf.strings.unicode_decode(t, "UTF8")
# print(t)

# ragged tensor
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
# # index op
# print(r)
# print(r[1])
# print(r[1:2])

# ops on ragged tensor
# r2 = tf.ragged.constant([[51, 52], [], [71]])
# print(tf.concat([r, r2], axis=0))
# r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43]])
# print(tf.concat([r, r3], axis=1))
# print(r.to_tensor())

# sparse tensor
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
# print(s)
# print(tf.sparse.to_dense(s))

# ops on sparse tensors
s2 = s * 2.0
# print(s2)
# try:
#     s3 = s + 1
# except TypeError as ex:
#     print(ex)

# s4 = tf.constant([[10., 20.],
#                   [30., 40.],
#                   [50., 60.],
#                   [70., 80.]])
# print(tf.sparse.sparse_dense_matmul(s, s4))
# s5 = tf.SparseTensor(indices=[[0, 2], [0, 1], [2, 3]],
#                      values=[1., 2., 3.],
#                      dense_shape=[3, 4])
# print(s5)
# s6 = tf.sparse.reorder(s5)
# print(tf.sparse.to_dense(s6))


# Variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
# print(v)
# print(v.value())
# print(v.numpy())

# assgin value重新赋值
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())


