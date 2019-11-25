from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# 定义模型
model = Sequential()
model.add(Conv2D(input_shape=(150, 150, 3), filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=20,  # 随机错切变换
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255,  # 数据归一化
)

batch_size = 32

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    'image/train',
    target_size=(150, 150),
    batch_size=batch_size,
)

# 测试数据
test_generator = test_datagen.flow_from_directory(
    'image/test',
    target_size=(150, 150),
    batch_size=batch_size,
)
print(train_generator.class_indices)
# 训练模型
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=30,
                    validation_data=test_generator,
                    validation_steps=len(test_generator))

# pip install h5py
model.save('model_cnn.h5')
