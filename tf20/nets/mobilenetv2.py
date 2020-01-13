from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D, ReLU
from tensorflow.keras import backend as K


def block(inputs, out_chans, k, s):
    x = Conv2D(out_chans, k, padding='same', strides=s)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)
    return x


def inverted_residual_block(inputs, out_chans, k, t, s, r=False):
    # Depth
    tchannel = K.int_shape(inputs)[-1] * t

    x = block(inputs, tchannel, 1, 1)
    # 注意这里depth_multiplier这个参数的含义。
    x = DepthwiseConv2D(k, strides=s, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    x = Conv2D(out_chans, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    # 此处不使用ReLU操作
    if r:
        x = Add()([x, inputs])

    return x


def make_layer(inputs, out_chans, k, t, s, n):
    x = inverted_residual_block(inputs, out_chans, k, t, s)

    for i in range(1, n):
        x = inverted_residual_block(x, out_chans, k, t, 1, True)

    return x


def MobileNetv2(input_shape, k):
    inputs = Input(shape=input_shape)

    x = block(inputs, 32, (3, 3), s=(2, 2))
    x = inverted_residual_block(x, 16, (3, 3), t=1, s=1)

    x = make_layer(x, 24, (3, 3), t=6, s=2, n=2)
    x = make_layer(x, 32, (3, 3), t=6, s=2, n=3)
    x = make_layer(x, 64, (3, 3), t=6, s=2, n=4)
    x = make_layer(x, 96, (3, 3), t=6, s=1, n=3)
    x = make_layer(x, 160, (3, 3), t=6, s=2, n=3)
    x = inverted_residual_block(x, 320, (3, 3), t=6, s=1)

    x = block(x, 1280, (1, 1), s=(1, 1))  # [n,7,7,c]
    x = GlobalAveragePooling2D()(x)  # [n,c]
    x = Reshape((1, 1, 1280))(x)  # [n,1,1,1280]
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)  # [n,1,1,k]

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)

    return model


if __name__ == "__main__":
    net = MobileNetv2((224, 224, 3), 100)
    net.summary()
