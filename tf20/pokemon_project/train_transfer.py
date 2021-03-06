import matplotlib
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
from pokemon import load_pokemon, normalize

tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def preprocess(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [224, 224])

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)
    return x, y


if __name__ == '__main__':
    batch_size = 32
    root_dir = "F:\\data\\pokemon"
    images, labels, table = load_pokemon(root_dir, mode='train')
    db_train = tf.data.Dataset.from_tensor_slices((images, labels))
    db_train = db_train.shuffle(1000).map(preprocess).batch(batch_size)

    images2, labels2, table = load_pokemon(root_dir, mode='val')
    db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
    db_val = db_val.map(preprocess).batch(batch_size)

    images3, labels3, table = load_pokemon(root_dir, mode='test')
    db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
    db_test = db_test.map(preprocess).batch(batch_size)

    net = keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='max')
    net.trainable = False

    newnet = keras.Sequential([
        net,
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.5),
        layers.Dense(5)
    ])
    newnet.build(input_shape=(1, 224, 224, 3))
    newnet.summary()

    # early_stoping = EarlyStopping(
    #     monitor='val_accuracy',
    #     min_delta=0.001,
    #     patience=3
    # )
    # newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
    #                loss=losses.CategoricalCrossentropy(from_logits=True),
    #                metrics=['accuracy'])
    # history = newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100, callbacks=[early_stoping])
    # history = history.history
    # print(history.keys())
    # print(history['val_accuracy'])
    # print(history['accuracy'])
    #
    # test_acc = newnet.evaluate(db_test)
    # plt.figure()
    # returns = history['val_accuracy']
    # plt.plot(np.arange(len(returns)), returns, lable='验证准确率')
    # plt.plot(np.arange(len(returns)), returns, 's')
    # returns = history['accuracy']
    # plt.plot(np.arange(len(returns)), returns, label='训练准确率')
    # plt.plot(np.arange(len(returns)), returns, 's')
    # plt.plot([len(returns) - 1], [test_acc[-1]], 'D', label='测试准确率')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('准确率')
    # plt.savefig('transfer.svg')
