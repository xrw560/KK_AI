from tensorflow import keras

model = keras.models.Sequential([
    # keras.layers.Dense(128 * 5 * 5, input_dim=100),
    # keras.layers.Reshape((5, 5, 128)),
    keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(3, 3), input_shape=(224, 224, 1))
])
model.summary()

# X = np.array([
#     [1, 2],
#     [3, 4]
# ])
# X = X.reshape([1, 2, 2, 1])
#
# model = keras.Sequential([
#     # keras.layers.UpSampling2D(input_shape=(2, 2, 1))
#     keras.layers.Conv2DTranspose(1, (1, 1), strides=(2, 2), input_shape=(2, 2, 1))
# ])
# model.summary()
#
# weights = [np.asarray([[[[1]]]]), np.asarray([0])]
# model.set_weights(weights=weights)
# y_hat = model.predict(X)
# y_hat = y_hat.reshape((4, 4))
# print(y_hat)
#
