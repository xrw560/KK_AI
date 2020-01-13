import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import os
import subprocess  # 进程管理

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("\ntrain_images.shape:{}, of {}".format(train_images.shape, train_images.dtype))
print("test_images.shape:{}, of {}".format(test_images.shape, test_images.dtype))

model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax', name='softmax')
])
model.summary()

testing = False
epochs = 5
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

import tempfile

MODEL_DIR = os.getcwd()
version = 1
export_path = os.path.join(MODEL_DIR, str(version) + ".h5")
print('export_path={}\n'.format(export_path))

model.save(export_path)
