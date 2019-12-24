## tf2.0
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
## tf2.0-keras
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

#全链接层模型
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

