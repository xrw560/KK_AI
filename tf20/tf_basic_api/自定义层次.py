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

### 房价数据
from sklearn.datasets import fetch_california_housing

# layer = tf.keras.layers.Dense(100, input_shape=(None, 5))
# print(layer(tf.zeros([10, 5])))
# print(layer.variables)
# print(help(layer))

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.25
)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# tf.nn.softplus: log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 5., 10.]))

#
# customized dense layer
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建所需要的参数"""
        # x * w + b
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完整正向计算"""
        return self.activation(x @ self.kernel + self.bias)


model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus
])

model.summary()

# model.compile(loss="mean_squared_error", optimizer='sgd')  # 验证是否一致
# callbacks = [
#     keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
# ]
# history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=10, callbacks=callbacks)
#
#
# def plot_learning_curves(history):
#     pd.DataFrame(history.history).plot(figsize=(8, 5))
#     plt.grid(True)
#     plt.gca().set_ylim(0, 1)
#     plt.show()
#
#
# plot_learning_curves(history)
# print(model.evaluate(x_test_scaled, y_test))
