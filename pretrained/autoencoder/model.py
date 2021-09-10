import tensorflow as tf
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers

from loss import contrastive_loss, cosine_similarity
from layer import conv1, conv2, tran2, drop, kwargs
from config import INPUT_SHAPE

        
def autoencoder(p):
    inp = Input(INPUT_SHAPE)
    
    # --- Encoder
    e1 = drop(conv1(4, inp), p['drop_rate'])
    e2 = drop(conv1(8, conv2(8, e1)), p['drop_rate'])
    e3 = drop(conv1(16, conv2(16, e2)), p['drop_rate'])
   
    e4 = drop(conv1(32, conv2(32, e3)), p['drop_rate'])
    
    # --- Embedding
    e5 = layers.Conv3D(filters=4, kernel_size=(1, 1, 1))(e4)
    e6 = layers.Flatten()(e5)
    e7 = layers.Dense(10, activation="relu", name="ctr")(e6)
    e8 = layers.Dense(1, activation="sigmoid", name="enc")(e7)

    # --- Decoder
    d1 = tran2(16, e4)
    d2 = conv1(8, tran2(8, d1))
    d3 = conv1(4, tran2(8, d2))
    d4 = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), name="dec")(d3)

    autoencoder_logits = {}
    autoencoder_logits["ctr"] = e7
    autoencoder_logits["enc"] = e8
    autoencoder_logits["dec"] = d4

    return Model(inputs=inp, outputs=autoencoder_logits)

    
def prepare_model(inputs, p, use_cosine_similarity=True):
    
    # --- Define autoencoder network
    autoencoder_network = autoencoder(p)

    # --- Define contrastive network
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
    
    siamese_logits = {}
    siamese_logits["ctr1"] = layers.Dense(1, activation="sigmoid", name="ctr1")(merge_layer1)
    siamese_logits["ctr2"] = layers.Dense(1, activation="sigmoid", name="ctr2")(merge_layer2)
    siamese_logits["ctr3"] = layers.Dense(1, activation="sigmoid", name="ctr3")(merge_layer3)
    siamese_logits["enc1"] = layers.Layer(name="enc1")(tower_1["enc"])
    siamese_logits["enc2"] = layers.Layer(name="enc2")(tower_2["enc"])
    siamese_logits["enc3"] = layers.Layer(name="enc3")(tower_3["enc"])
    siamese_logits["dec1"] = layers.Layer(name="dec1")(tower_1["dec"])
    siamese_logits["dec2"] = layers.Layer(name="dec2")(tower_2["dec"])
    siamese_logits["dec3"] = layers.Layer(name="dec3")(tower_3["dec"])
    
    siamese = Model(inputs=inputs, outputs=siamese_logits)
    
    siamese.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']),
        loss={
            'ctr1': contrastive_loss(),
            'ctr2': contrastive_loss(),
            'ctr3': contrastive_loss(),
            'dec1': losses.MeanAbsoluteError(),
            'dec2': losses.MeanAbsoluteError(),
            'dec3': losses.MeanAbsoluteError(),
            'enc1': losses.BinaryCrossentropy(),
            'enc2': losses.BinaryCrossentropy(),
            'enc3': losses.BinaryCrossentropy()
        },
        loss_weights={
            'ctr1': 1*p['contrast'],
            'ctr2': 1*p['contrast'],
            'ctr3': 1*p['contrast'],
            'dec1': 1*p['recon'],
            'dec2': 1*p['recon'],
            'dec3': 1*p['recon'],
            'enc1': 1,
            'enc2': 1,
            'enc3': 1,
        },
        metrics={
            'enc1': metrics.BinaryAccuracy(),
            'enc2': metrics.BinaryAccuracy(),
            'enc3': metrics.BinaryAccuracy()
        },
        experimental_run_tf_function=False
    )
    
    return siamese