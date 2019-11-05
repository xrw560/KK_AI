import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 转换one-hot
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential([
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
])

# 定义优化器
sgd = SGD(lr=0.2)

model.compile(
    optimizer=sgd,
    loss='mse',
    metrics=['accuracy']
)
# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print("\ntest loss:", loss)
print("accuracy", accuracy)
