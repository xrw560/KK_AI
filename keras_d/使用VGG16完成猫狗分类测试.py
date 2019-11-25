from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

label = np.array(['cat', 'dog'])
# 载入模型
model = load_model('model_vgg16.h5')

# 导入图片
image = load_img('image/test/cat/cat.1003.jpg')
image = image.resize((150, 150))
image = img_to_array(image)
image = image / 255
image = np.expand_dims(image, 0)
print(image.shape)
print(label[model.predict_classes(image)])
