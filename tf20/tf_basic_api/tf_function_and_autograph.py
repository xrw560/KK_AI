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
import timeit


# def scaled_elu(z, scale=1.0, alpha=1.0):
#     # z > 0 ? scaled * z : scaled * alpha * tf.nn.elu(z)
#     is_positive = tf.greater_equal(z, 0.0)
#     return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
#
#
# def display_tf_code(func):
#     code = tf.autograph.to_code(func)
#     from IPython.display import display, Markdown
#     display(Markdown('```python\n{}\n```'.format(code)))
#
# display_tf_code(scaled_elu)

# # 1 + 1/2 + 1/2^2 + ... + 1/2^n
# @tf.function
# def converge_to_2(n_iters):
#     total = tf.constant(0.)
#     increment = tf.constant(1.)
#     for _ in range(n_iters):
#         total += increment
#         increment /= 2.0
#     return total
#
# print(converge_to_2(20))

# tf.function and auto-graph
# def scaled_elu(z, scale=1.0, alpha=1.0):
#     # z > 0 ? scaled * z : scaled * alpha * tf.nn.elu(z)
#     is_positive = tf.greater_equal(z, 0.0)
#     return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
# #
# # print(scaled_elu(tf.constant(-3.)))
# # print(scaled_elu(tf.constant([-3., -2.5])))
# #
# # scaled_elu_tf = tf.function(scaled_elu)
# # print(scaled_elu_tf(tf.constant(-3.0)))
# # print(scaled_elu_tf(tf.constant([-3., -2.5])))
# #
# # print(scaled_elu_tf.python_function is scaled_elu)
#
#
# foo1 = """
# import tensorflow as tf
# def scaled_elu(z, scale=1.0, alpha=1.0):
#     # z > 0 ? scaled * z : scaled * alpha * tf.nn.elu(z)
#     is_positive = tf.greater_equal(z, 0.0)
#     return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
# scaled_elu(tf.random.normal((1000,1000)))
# """
#
# foo2 = """
# import tensorflow as tf
# def scaled_elu(z, scale=1.0, alpha=1.0):
#     # z > 0 ? scaled * z : scaled * alpha * tf.nn.elu(z)
#     is_positive = tf.greater_equal(z, 0.0)
#     return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
# scaled_elu_tf = tf.function(scaled_elu)
# scaled_elu_tf(tf.random.normal((1000,1000)))
# """
# print(timeit.timeit(stmt=foo1, number=1))
# print(timeit.timeit(stmt=foo2, number=1))
#
# var = tf.Variable(0.)
#
# @tf.function
# def add_21():
#     return var.assign_add(21)
# print(add_21)

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)


# try:
#     print(cube(tf.constant([1., 2., 3.])))
# except ValueError as ex:
#     print(ex)
# print(cube(tf.constant([1, 2, 3])))

# @tf.function py func -> tf graph
# get_concrete_function -> add input signature -> SavedModel

cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
# print(cube_func_int32)
# print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
# print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))
# print(cube_func_int32.graph)
# print(cube_func_int32.graph.get_operations())
pow_op = cube_func_int32.graph.get_operations()[2]
# print(pow_op)
# print(list(pow_op.inputs))
# print(list(pow_op.outputs))
# print(cube_func_int32.graph.get_operation_by_name("x"))
# print(cube_func_int32.graph.get_tensor_by_name('x:0'))
print(cube_func_int32.graph.as_graph_def())