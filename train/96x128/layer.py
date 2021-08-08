from tensorflow.keras import layers


# --- Define lambda functions
kwargs = {
    'kernel_size': (3, 3, 3),
    'padding': 'same',
    'kernel_initializer': 'he_uniform'
}

conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
norm = lambda x : layers.BatchNormalization()(x)
acti = lambda x : layers.LeakyReLU()(x)
tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)

conv1 = lambda filters, x : norm(acti(conv(x, filters, strides=1)))
conv2 = lambda filters, x : norm(acti(conv(x, filters, strides=(2, 2, 2))))
tran2 = lambda filters, x : norm(acti(tran(x, filters, strides=(2, 2, 2))))