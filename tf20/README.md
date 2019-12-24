## tf 2.0
```python
import tensorflow as tf
tf.__version__

!pip install tensorflow

#张量
#常量
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)
print(x)
print(a+b)

a.get_shape()

a.numpy()

#变量
s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([784,10]))
s.assign(3)
s.assign_add(3)

class MyModule(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]

m = MyModule()
m.variables

#tf.data
dataset = tf.data.Dataset.from_tensors([1,2,3,4,5])
for element in dataset:
  print(element.numpy())
  it = iter(dataset)
print(next(it).numpy())

dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
for element in dataset:
  print(element.numpy())
it = iter(dataset)
print(next(it).numpy())

features = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
labels = tf.data.Dataset.from_tensor_slices([6,7,8,9,10])
dataset = tf.data.Dataset.zip((features,labels))
for element in dataset3:
  print(element)

inc_dataset = tf.data.Dataset.range(100)   
dec_dataset = tf.data.Dataset.range(0, -100, -1) 
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)  

for batch in batched_dataset.take(4):
  print([arr.numpy() for arr in batch])

shuffle_dataset = dataset.shuffle(buffer_size=10)
for element in shuffle_dataset:
  print(element)

shuffle_dataset = dataset.shuffle(buffer_size=100)
for element in shuffle_dataset:
  print(element)
```
## tf 2.0-keras
```python
#读取模型
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#获得图片大小
train_images.shape
#打印图例
import numpy as np
import matplotlib.pyplot as plt
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(train_images[:5])

#归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

#全连接层模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', bias=False, trainable=False),
    tf.keras.layers.Dense(10, activation='softmax')
])
#模型总结
model.summary()
#编译
model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy', categorical_crossentropy->[1,0,0,0,0]
      metrics=['accuracy']) 
      
#训练
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
#模型权重
model.variables

# 保存权重
model.save_weights('./fashion_mnist/my_checkpoint')

# 恢复权重
model.load_weights('./fashion_mnist/my_checkpoint')
# model1.load_weights('./fashion_mnist/my_checkpoint') 

# 预测
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#保存整个模型
model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "fashion_mnist_1/cp-{epoch:04d}.ckpt"

# 创建一个回调，每个epoch保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True,
    period=1) #save_freq = ‘epoch'/n samples 60000 100 600

# 使用 `checkpoint_path` 格式保存权重
model.save_weights(checkpoint_path.format(epoch=0))

# 使用新的回调训练模型
model.fit(train_images, 
              train_labels,
              epochs=5, 
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels))
              
#CNN模型
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(tf.keras.layers.MaxPooling2D((2, 2)))
model1.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPooling2D((2, 2)))
model1.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(256, activation='relu'))
model1.add(tf.keras.layers.Dense(10, activation='softmax'))

model1.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
      
model1.fit(train_images, train_labels, 
           batchsize=64,
           epochs=10, validation_data=(test_images, test_labels))

#RNN
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.LSTM(128,input_shape=(None,28))) batchsize,28,28
# model2.add(tf.keras.layers.LSTM(128, return_sequences=True))
#(hidden size * (hidden size + input_dim ) + hidden size) *4
model2.add(tf.keras.layers.Dense(10, activation='softmax'))

model2.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

model2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```
## 自定义模型
初始化时生成w和b
```python
from tensorflow.keras import layers

#自定义线性层
class Linear(layers.Layer): 
  def __init__(self, units=32, input_dim=28):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),dtype='float32'),trainable=False)
    #self.w = self.add_weight(shape=(input_dim, units),initializer='random_normal',trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),dtype='float32'),trainable=False)

  def call(self, inputs): 
    return tf.matmul(inputs, self.w) + self.b

inputs = tf.ones((2, 2)) 
linear_layer = Linear(4, 2)
print(linear_layer.w)
outputs = linear_layer(inputs)
print(linear_layer.w)#调用call函数
```

```python
#input_dim未知
class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

inputs = tf.ones((2, 2))
linear_layer = Linear(4)
outputs = linear_layer(inputs) #此时才会生成w和b
print(linear_layer.w)
```

```python
#模型
class MLPBlock(layers.Layer):

  def __init__(self):
    super(MLPBlock, self).__init__()
    self.linear_1 = Linear(32)
    self.linear_2 = Linear(32)
    self.linear_3 = Linear(1)

  def call(self, inputs):
    x = self.linear_1(inputs)
    x = tf.nn.relu(x)
    x = self.linear_2(x)
    x = tf.nn.relu(x)
    return self.linear_3(x)

#训练
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for x_batch_train, y_batch_train in train_dataset:
  #取一个batch
  with tf.GradientTape() as tape:   
    logits = layer(x_batch_train)  
    loss_value = loss_fn(y_batch_train, logits)
  #？为什么在with外面
  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

## GPU模型

GPU支持
多GPU模型

硬件要求：
英伟达显卡
https://developer.nvidia.com/cuda-gpus
软件要求：
NVIDIA GPU驱动程序
https://www.nvidia.com/Download/index.aspx?lang=en-us
CUDA
对通用GPU计算操作做了封装，方便其他组件调用
https://developer.nvidia.com/cuda-zone



!pip install tensorflow-gpu

```python
#查看gpu
import tensorflow as tf
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

#查看可用的gpu数量
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)

#限制显存使用
import tensorflow as tf

# 设置显存使用上限
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7) 8G*0.7=5.6G 
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

# 按需取用
gpu_options = tf.GPUOptions(allow_growth=True)  4G  6G 
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#keras
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model

#cpu
with tf.device('/device:CPU:0'):
  model= Xception(weights=None, input_shape=(w,h,3), classes=100)

#单个gpu
with tf.device('/device:GPU:0'):
  model= Xception(weights=None, input_shape=(w,h,3), classes=100)

#多个gpu
model1=multi_gpu_model(model , gpus=8)
model1.compile(loss='crossentropy',optimizer='adam')

#训练
model1.fit(x,y,batch_size=256)
#单个gpu的batch_size=256/8=32


```

##  在移动设备上部署机器学习模型
在移动设备上部署机器学习模型

1.操作系统不同
2.存储容量不同

主要目的：减小模型文件大小，保持准确率

```python
#TensorFlow Lite 
#转化器 和 翻译器
import tensorflow as tf

#创建转化器（共三种方式）
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # 已保存好的模型
converter = tf.lite.TFLiteConverter.from_keras_model(model) # 定义的keras model
converter = tf.lite.TFLiteConverter.from_concrete_functions() # 
#转化模型
tflite_model = converter.convert()
#保存文件
open("converted_model.tflite", "wb").write(tflite_model)
```

```python
#在不同系统上，翻译器所用的语言不同
#安卓：Java or C++
#苹果：Swift or Objective-C

#加载并运行模型
import numpy as np
import tensorflow as tf

#创建翻译器
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

#为输入输出分配张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#输入
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
#model(input_data)
#计算
interpreter.invoke()
#输出
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### 量化模型

```python
#只量化权重
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#量化整个计算流程
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

```python
#量化输入输出
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.unint8
converter.inference_output_type=tf.unint8
```

```python
#特定量化
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types=[tf.lite.constants.FLOAT16]
```

```python
#量化模型
import tensorflow as tf
#创建转化器
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#优化类型
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#转换
tflite_quant_model = converter.convert()

```


## 图像分类项目
1. 数据预处理
2. 使用常用的模型/自己设计模型
3. 训练
4. 调参
5. 再训练
6. 总结经验


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#cifar100数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

#归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

print(len(train_images))
print(len(test_images))

dic = {}
for label in train_labels:
  x = int(label)
  if x not in dic:
    dic[x] = 1
  else:
    dic[x] += 1
print(dic)

print(train_images[0].shape)
print(train_images[100].shape)
print(train_images[10000].shape)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i][0])
plt.show()

#搭建模型
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))

#在小规模训练集上训练
train_images1 = train_images[0:100]
train_labels1 = train_labels[0:100]

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images1, train_labels1, epochs=100)
```

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
```

```python
#减小模型
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(100, activation='softmax'))

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history1 = model1.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history1.history['loss'], label='loss')
plt.plot(history1.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
```

```python
#正则化
model2 = models.Sequential()
model2.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model2.add(layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))

model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history2 = model2.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history2.history['loss'], label='loss')
plt.plot(history2.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#正则化，调参
model3 = models.Sequential()
model3.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
model3.add(layers.Flatten())
model3.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
model3.add(layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))

model3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history3 = model3.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history3.history['loss'], label='loss')
plt.plot(history3.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#正则化+减小参数
model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))

model4.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history4 = model4.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history4.history['loss'], label='loss')
plt.plot(history4.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#搭建模型，droupout
model5 = models.Sequential()
model5.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Dropout(0.2))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Dropout(0.2))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.Flatten())
model5.add(layers.Dense(512, activation='relu'))
model5.add(layers.Dense(100, activation='softmax'))

model5.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history5 = model5.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history5.history['loss'], label='loss')
plt.plot(history5.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#搭建模型，droupout
model6 = models.Sequential()
model6.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Dropout(0.1))
model6.add(layers.Conv2D(128, (3, 3), activation='relu'))
model6.add(layers.MaxPooling2D((2, 2)))
model6.add(layers.Dropout(0.1))
model6.add(layers.Conv2D(128, (3, 3), activation='relu'))
model6.add(layers.Flatten())
model6.add(layers.Dense(512, activation='relu'))
model6.add(layers.Dense(100, activation='softmax'))

model6.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history6 = model6.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
plt.plot(history6.history['loss'], label='loss')
plt.plot(history6.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#搭建模型，droupout
model7 = models.Sequential()
model7.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.Conv2D(128, (3, 3), activation='relu'))
model7.add(layers.MaxPooling2D((2, 2)))
model7.add(layers.Conv2D(128, (3, 3), activation='relu'))
model7.add(layers.Flatten())
model7.add(layers.Dense(512, activation='relu'))
model7.add(layers.Dropout(0.2))
model7.add(layers.Dense(100, activation='softmax'))

model7.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history7 = model7.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
                    plt.plot(history7.history['loss'], label='loss')
plt.plot(history7.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
​```

​```python
#正则化+减小参数
model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
model4.add(layers.Dense(100, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
```

```python
#调整learning_rate
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=20, initial_epoch=0
                    validation_data=(test_images, test_labels))
                    
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=40,
                    validation_data=(test_images, test_labels))
                    
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
                    
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=20, initial_epoch=0,validation_data=(test_images, test_labels))
```
```python
#增加步数
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=40,
                    validation_data=(test_images, test_labels))
                   
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model4.fit(train_images, train_labels, epochs=15,
                    validation_data=(test_images, test_labels))
# 训练一定步数之后，调小learning_rate
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history1 = model4.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))
                    
plt.plot(history.history['loss']+history1.history['loss'],label='loss')
plt.plot(history.history['val_loss']+history1.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
```

## 图像数据增强
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 64
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                 		  directory=train_dir
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
                                                           
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
                                                              
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
 plotImages(sample_training_images[:5])
 
 model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')#二分类
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

```python
#水平翻转
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
```

```python
#竖直翻转
image_gen = ImageDataGenerator(rescale=1./255, vertical_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
```

```python
#随机旋转
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45) -45,0,45
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
```

```python
# 随机缩放
# zoom_range from 0 - 1
# If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) # 
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
```

```python
#全部应用
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=0.5
                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')
                                                     
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
                    

```

## 迁移学习

迁移学习

使用别人已经训练好的模型来训练自己的任务

速度快，效果好

两种方法:

1. 加层：在别人的模型后面加几层，用来学习我们的任务

2. fine tune：不改变网络，最后几层重新训练

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

#可选模型：densenet、inception、mobilenet、resnet、VGG
base_model = tf.keras.applications.ResNet50(weights='imagenet')

base_model.summary()

base_model.trainable = False # 不训练base_model

batch_size = 64
epochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                              directory=train_dir,
                              shuffle=True,
                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                              class_mode='binary')
                              
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 3, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                directory=validation_dir,
                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                               class_mode='binary')
                             
sample_testing_images, _ = next(val_data_gen)
plotImages(sample_testing_images[:3])

prediction_layer1 = tf.keras.layers.Dense(128,activation='relu')
prediction_layer2 = tf.keras.layers.Dense(1,activation='sigmoid')

model = tf.keras.Sequential([
  base_model,
  prediction_layer1,
  prediction_layer2
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
history = model.fit_generator(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```
### fine tune
```python
# fine tune
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 150

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  
base_model.summary()

prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')
model = tf.keras.Sequential([
  base_model,
  prediction_layer
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(
    train_data_gen,
    epochs=epochs*2,
    validation_data=val_data_gen
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

## 词向量
词向量

1. one hot
   the 100
   cat 010
   sat 001

2. unique number
the 1
cat 2
sat 3
3. Word embeddings
the 0.8 1.0 2.2
cat 1.2 -0.1 4.3
sat 0.4 2.5  -0.9
```python
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

embedding_layer = layers.Embedding(1000, 5)
#和全链接层一样吗

result = embedding_layer(tf.constant([1,2,3]))
result.numpy()

#result = embedding_layer(tf.constant([1230]))

result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
result.shape

#https://www.tensorflow.org/datasets/catalog/imdb_reviews
train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)
    
encoder = info.features['text'].encoder
print(len(encoder.subwords))

encoder.subwordsa[0:20]

padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)

train_batch, train_labels = next(iter(train_batches))
print(train_batch.shape)
train_batch.numpy()

train_batch, train_labels = next(iter(train_batches))
print(train_batch.shape)
train_batch.numpy()

train_labels

#建模分析情感
embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(5,5))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) 

import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
  vec = weights[num+1] # skip 0, it's padding.
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

```


## 风格迁移
```python
#风格转移

#‘换脸’

#原理：
#生成新的图片
#1.内容与原图接近
#2.风格与要求图片接近
#
#对于CNN
#较深层与内容相关
#较浅层与风格相关

from __future__ import absolute_import, division, print_function, unicode_literals
#安装最新版tensorflow
!pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
#下载图片或加载本地图片
content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools

#加载图片，画图演示
contentimg = tf.io.read_file(content_path)
contentimg = tf.image.decode_image(contentimg, channels=3)
contentimg = tf.image.convert_image_dtype(contentimg, tf.float32)
plt.subplot(1, 2, 1)
plt.imshow(contentimg)
contentimg = contentimg[tf.newaxis, :]

styleimg = tf.io.read_file(style_path)
styleimg = tf.image.decode_image(styleimg, channels=3)
styleimg = tf.image.convert_image_dtype(styleimg, tf.float32)
plt.subplot(1, 2, 2)
plt.imshow(styleimg)
styleimg = styleimg[tf.newaxis, :]

#加载vgg模型
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

for layer in vgg.layers:
  print(layer.name)
  
# 内容层
content_layers = ['block5_conv2'] 

# 风格层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
#尝试一下使用不同的风格层和内容层
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

#建立模型输出中间层
def vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

#计算风格参数
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

#输出风格和内容参数
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}

#提取特征
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(contentimg))

style_results = results['style']
#输出提取到的特征
print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  
#优化
style_targets = extractor(styleimg)['style']
content_targets = extractor(contentimg)['content']

image = tf.Variable(contentimg)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
#优化器
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
#内容和风格用不同的系数
style_weight=1e-2
content_weight=1e4
#loss 函数
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

#更新图片
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

#训练
for i in range(5):
    train_step(image)

#输出对比
plt.subplot(1, 2, 1)
plt.imshow(image.read_value()[0])

contentimg = tf.io.read_file(content_path)
contentimg = tf.image.decode_image(contentimg, channels=3)
contentimg = tf.image.convert_image_dtype(contentimg, tf.float32)
plt.subplot(1, 2, 2)
plt.imshow(contentimg)
contentimg = contentimg[tf.newaxis, :]

```
