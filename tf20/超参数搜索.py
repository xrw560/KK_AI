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

# learning_rate: [1e-4, 3e-4, 1e-3, 3e-3,1e-2,3e-2]
# W = W + grad * learning_rate
learning_rate = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
historys = []
for lr in learning_rate:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]
    history = model.fit(x_train_scaled,
                        y_train,
                        validation_data=(x_valid_scaled, y_valid),
                        epochs=10,
                        callbacks=callbacks)
    historys.append(history)


def plot_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


for lr, history in zip(learning_rate, historys):
    print("Learning rate:", lr)
    plot_learning_curve(history)
