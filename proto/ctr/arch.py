from tensorflow.keras import Input, Model, layers
from jarvis.train import params

def ctr_cls_v01(inputs, use_cosine_similarity=True):
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
    
    # --- Encoder block
    inp = Input(INPUT_SHAPE)
    e1 = conv1(4, inp)
    e2 = conv1(8, conv2(8, e1))
    e3 = conv1(16, conv2(16, e2))
    e4 = conv1(32, conv2(32, e3))
    e5 = conv1(64, conv2(64, e4))
    e6 = conv1(128, conv2(128, e5))
    
    # --- Contrastive block
    bottleneck = layers.Conv3D(filters=4, activation='tanh', kernel_size=(1, 1, 1))(e6)
    latent_vec1 = layers.Flatten()(bottleneck)    
    latent_vec2 = layers.Dense(32, activation='tanh')(latent_vec1)
    
    ctr_out = layers.Dense(8, activation='tanh', name='ctr')(latent_vec2)
    enc_out = layers.Dense(1, activation='sigmoid', name='enc')(latent_vec2)
    
    autoencoder_logits = {}
    autoencoder_logits["ctr"] = ctr_out
    autoencoder_logits["enc"] = enc_out
    
    autoencoder_network = Model(inputs=inp, outputs=autoencoder_logits)

    tower_1 = autoencoder_network(inputs=inputs["pos"])
    tower_2 = autoencoder_network(inputs=inputs["unk"])
    tower_3 = autoencoder_network(inputs=inputs["neg"])
    
    if use_cosine_similarity:
        merge_layer1 = layers.Lambda(cosine_similarity)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(cosine_similarity)([tower_2["ctr"], tower_3["ctr"]])
        merge_layer3 = layers.Lambda(cosine_similarity)([tower_1["ctr"], tower_3["ctr"]])
    else:
        merge_layer1 = layers.Lambda(euclidean_distance)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(euclidean_distance)([tower_2["ctr"], tower_3["ctr"]])
        merge_layer3 = layers.Lambda(euclidean_distance)([tower_1["ctr"], tower_3["ctr"]])
    
    contrastive_logits = {}
    contrastive_logits["ctr1"] = layers.Layer(name="ctr1")(merge_layer1)
    contrastive_logits["ctr2"] = layers.Layer(name="ctr2")(merge_layer2)
    contrastive_logits["ctr3"] = layers.Layer(name="ctr3")(merge_layer3)
    contrastive_logits["enc1"] = layers.Layer(name="enc1")(tower_1["enc"])
    contrastive_logits["enc2"] = layers.Layer(name="enc2")(tower_2["enc"])
    contrastive_logits["enc3"] = layers.Layer(name="enc3")(tower_3["enc"])
    
    contrastive_network = Model(inputs=inputs, outputs=contrastive_logits)
    
    contrastive_network.compile(
        optimizer=optimizers.Adam(learning_rate=p['learning_rate']),
        loss={
            'ctr1': contrastive_loss(),
            'ctr2': contrastive_loss(),
            'ctr3': contrastive_loss(),
            'enc1': losses.BinaryCrossentropy(),
            'enc2': losses.BinaryCrossentropy(),
            'enc3': losses.BinaryCrossentropy()
        },
        loss_weights={
            'ctr1': p['ctr1'],
            'ctr2': p['ctr2'],
            'ctr3': p['ctr3'],
            'enc1': p['enc1'],
            'enc2': p['enc2'],
            'enc3': p['enc3']
        },
        metrics={
            'enc1': metrics.BinaryAccuracy(),
            'enc2': metrics.BinaryAccuracy(),
            'enc3': metrics.BinaryAccuracy(),
        },
        experimental_run_tf_function=False
    )
    
    return contrastive_network