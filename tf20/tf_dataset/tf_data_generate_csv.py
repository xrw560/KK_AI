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

output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, 'wt', encoding='utf-8') as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join([repr(col) for col in data[row_index]]))
                f.write("\n")
    return filenames


train_data = np.c_[x_train_scaled, y_train]  # 把数据按行Merge
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]

header_cols = housing.feature_names + ['MidianHouseValue']
header_str = "".join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, "train", header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, "valid", header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, "test", header_str, n_parts=10)

filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv

def csv_reader_dataset(filename, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filename)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='sgd')
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
history = model.fit(train_set,
                    validation_data=valid_set,
                    steps_per_epoch=11160 // batch_size,
                    validation_steps=3870 // batch_size,
                    epochs=10,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
model.evaluate(test_set, steps=5160 // batch_size)
