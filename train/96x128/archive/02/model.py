import os, random, time, numpy as np, pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, datasets, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload, tools as jtools

# --- Define path of clients
AD_CLIENT_PATH = '/data/raw/adni/data/ymls/client-3d-96x128_AD_only.yml'
CN_CLIENT_PATH = '/data/raw/adni/data/ymls/client-3d-96x128_CN_only.yml'
INPUT_SHAPE = (96, 128, 128, 1)

def contrastive_generator(valid=False):
    # --- Create generators for AD/CN 
    client_AD = Client(AD_CLIENT_PATH, configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}})
    client_CN = Client(CN_CLIENT_PATH, configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}})
    
    gen_train_AD, gen_valid_AD = client_AD.create_generators()
    gen_train_CN, gen_valid_CN = client_CN.create_generators()
    
    while True:
        if valid:
            xs_AD, ys_AD = next(gen_valid_AD)
            xs_CN, ys_CN = next(gen_valid_CN)
        else:
            xs_AD, ys_AD = next(gen_train_AD)
            xs_CN, ys_CN = next(gen_train_CN)
        
        # --- Randomize for AD-AD-CN or AD-CN-CN
        choice_index = random.randint(0, 1)
        
        if choice_index == 0:
            xs_final = np.concatenate((xs_AD['dat'], xs_CN['dat'][:1]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'], ys_CN['lbl'][:1]), axis=0)
        else:
            xs_final = np.concatenate((xs_AD['dat'][:1], xs_CN['dat']), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:1], ys_CN['lbl']), axis=0)

        xs = {}
        ys = {}
        
        xs['pos'] = np.expand_dims(xs_final[0], axis=0)
        xs['unk'] = np.expand_dims(xs_final[1], axis=0)
        xs['neg'] = np.expand_dims(xs_final[2], axis=0)
        ys['enc1'] = ys_final[0].reshape((1))
        ys['enc2'] = ys_final[1].reshape((1))
        ys['enc3'] = ys_final[2].reshape((1))
        ys['dec1'] = np.expand_dims(xs_final[0], axis=0)
        ys['dec2'] = np.expand_dims(xs_final[1], axis=0)
        ys['dec3'] = np.expand_dims(xs_final[2], axis=0)
        ys['ctr1'] = ys['enc1'] == ys['enc2']
        ys['ctr2'] = ys['enc2'] == ys['enc3']
            
        yield xs, ys
        
def cosine_similarity(vects):
    """Find the cosine similarity between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing cosine similarity
        (as floating point value) between vectors.
    """
    a, b = vects
    return 1 - tf.keras.layers.Dot(axes=1, normalize=True)([a, b])

def euclidean_distance2(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    a, b = vects
    return tf.norm(a - b, ord='euclidean')

def norm_euclidean_distance(vects):
    """Find the normalized Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing normalized euclidean distance
        (as floating point value) between vectors.
    """
    a, b = vects
    return tf.norm(tf.nn.l2_normalize(a, 0) - tf.nn.l2_normalize(b, 0), ord='euclidean')

def contrastive_loss(margin=1):
    """Provides 'ctr_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar (default is 1). The
                margin should correspond to the range of the distance function
                used to compare the latent vectors.

    Returns:
        'ctr_loss' function with data ('margin') attached.

    Resource:
        https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
    """
    
    def ctr_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum((margin - y_pred), 0))
        return tf.math.reduce_mean(
            y_true * square_pred + (1 - y_true) * margin_square
        )

    return ctr_loss

def prepare_model(inputs, use_cosine_similarity=True):
        
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
    
    # --- Define autoencoder network
    
    inp = Input(INPUT_SHAPE)
    e1 = conv1(4, inp)
    e2 = conv1(8, conv2(8, e1))
    e3 = conv1(16, conv2(16, e2))
    e4 = conv1(32, conv2(32, e3))
    e5 = layers.Conv3D(filters=4, kernel_size=(1, 1, 1))(e4)
    e6 = layers.Flatten()(e5)
    e7 = layers.Dense(10, activation="relu", name="ctr")(e6)
    e8 = layers.Dense(1, activation="sigmoid", name="enc")(e7)
    d1 = tran2(16, e4)
    d2 = conv1(8, tran2(8, d1))
    d3 = conv1(4, tran2(8, d2))
    d4 = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), name="dec")(d3)
    
    autoencoder_logits = {}
    autoencoder_logits["ctr"] = e7
    autoencoder_logits["enc"] = e8
    autoencoder_logits["dec"] = d4
    
    autoencoder_network = Model(inputs=inp, outputs=autoencoder_logits)
    
    # --- Define contrastive network

    tower_1 = autoencoder_network(inputs=inputs["pos"])
    tower_2 = autoencoder_network(inputs=inputs["unk"])
    tower_3 = autoencoder_network(inputs=inputs["neg"])
    
    if use_cosine_similarity:
        merge_layer1 = layers.Lambda(cosine_similarity)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(cosine_similarity)([tower_2["ctr"], tower_3["ctr"]])
    else:
        merge_layer1 = layers.Lambda(euclidean_distance)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(euclidean_distance)([tower_2["ctr"], tower_3["ctr"]])
    
    siamese_logits = {}
    siamese_logits["ctr1"] = layers.Dense(1, activation="sigmoid", name="ctr1")(merge_layer1)
    siamese_logits["ctr2"] = layers.Dense(1, activation="sigmoid", name="ctr2")(merge_layer2)
    siamese_logits["enc1"] = layers.Layer(name="enc1")(tower_1["enc"])
    siamese_logits["enc2"] = layers.Layer(name="enc2")(tower_2["enc"])
    siamese_logits["enc3"] = layers.Layer(name="enc3")(tower_3["enc"])
    siamese_logits["dec1"] = layers.Layer(name="dec1")(tower_1["dec"])
    siamese_logits["dec2"] = layers.Layer(name="dec2")(tower_2["dec"])
    siamese_logits["dec3"] = layers.Layer(name="dec3")(tower_3["dec"])
    
    siamese = Model(inputs=inputs, outputs=siamese_logits)
    
    siamese.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'ctr1': contrastive_loss(),
            'ctr2': contrastive_loss(),
            'dec1': losses.MeanSquaredError(),
            'dec2': losses.MeanSquaredError(),
            'dec3': losses.MeanSquaredError(),
            'enc1': losses.BinaryCrossentropy(),
            'enc2': losses.BinaryCrossentropy(),
            'enc3': losses.BinaryCrossentropy()
        },
        loss_weights={
            'ctr1': 0.5,
            'ctr2': 0.5,
            'dec1': 0.5,
            'dec2': 0.5,
            'dec3': 0.5,
            'enc1': 0.5,
            'enc2': 0.5,
            'enc3': 0.5
        },
        metrics={
            'enc1': metrics.BinaryAccuracy(),
            'enc2': metrics.BinaryAccuracy(),
            'enc3': metrics.BinaryAccuracy()
        },
        experimental_run_tf_function=False
    )
    
    return siamese

# --- Autoselect GPU
gpus.autoselect()

# --- Prepare hyperparams
p = params.load('./hyper.csv', row=0)

MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- Prepare model
inputs = {
    'pos': Input(shape=INPUT_SHAPE, name='pos'),
    'unk': Input(shape=INPUT_SHAPE, name='unk'),
    'neg': Input(shape=INPUT_SHAPE, name='neg'),
}

gen_train = contrastive_generator()
gen_valid = contrastive_generator(valid=True)

model = prepare_model(inputs, use_cosine_similarity=True)

# --- Set training variables
steps_per_epoch = 100
validation_freq = 1

# --- Determine total loop iterations needed
epochs = int(p['iterations'] / steps_per_epoch)

# --- Prepare Tensorboard 
log_dir = '{}/jmodels/logdirs/{}'.format(
    os.path.dirname(p['output_dir']),
    os.path.basename(p['output_dir']))

# --- Prepare CSV path
csv_log_path = '{}/train_log.csv'.format(p['output_dir'])

# --- Define training callbacks
tensorboard_log = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
csv_log = callbacks.CSVLogger(csv_log_path)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# --- Train the model
model.fit(
    x=gen_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=gen_valid,
    validation_steps=steps_per_epoch,
    validation_freq=validation_freq,
    callbacks=[tensorboard_log, csv_log]
)

# --- Save model
model.save(MODEL_NAME)