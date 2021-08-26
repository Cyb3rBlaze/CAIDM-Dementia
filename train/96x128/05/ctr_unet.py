import os, random, time, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload

# --- Define path of clients
AD_CLIENT_PATH = '/home/mmorelan/proj/dementia/yml/client-3d-96x128_AD_AV45_only.yml'
CN_CLIENT_PATH = '/home/mmorelan/proj/dementia/yml/client-3d-96x128_CN_AV45_only.yml'
INPUT_SHAPE = (96, 128, 128, 1)

@overload(Client)
def preprocess(self, arrays, **kwargs):
    
    # --- Extract pre-calculated whole exam mu/sd and normalize
    arrays['xs']['dat'] = (arrays['xs']['dat'] - kwargs['row']['mu']) / kwargs['row']['sd']
    
    # --- Scale to 0/1 using 5/95 percentiles
    lower = np.percentile(arrays['xs']['dat'], 1)
    upper = np.percentile(arrays['xs']['dat'], 99)
    arrays['xs']['dat'] = arrays['xs']['dat'].clip(min=lower, max=upper)
    arrays['xs']['dat'] = (arrays['xs']['dat'] - lower) / (upper - lower)
    
    return arrays

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
            xs_final = np.concatenate((xs_AD['dat'][:2], xs_CN['dat'][:1]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:2], ys_CN['lbl'][:1]), axis=0)
        else:
            xs_final = np.concatenate((xs_AD['dat'][:1], xs_CN['dat'][:2]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:1], ys_CN['lbl'][:2]), axis=0)

        xs = {}
        ys = {}
        
        xs['pos'] = np.expand_dims(xs_final[0], axis=0)
        xs['unk'] = np.expand_dims(xs_final[1], axis=0)
        xs['neg'] = np.expand_dims(xs_final[2], axis=0)
        ys['cls1'] = ys_final[0].reshape((1))
        ys['cls2'] = ys_final[1].reshape((1))
        ys['cls3'] = ys_final[2].reshape((1))
        ys['dec1'] = np.expand_dims(xs_final[0], axis=0)
        ys['dec2'] = np.expand_dims(xs_final[1], axis=0)
        ys['dec3'] = np.expand_dims(xs_final[2], axis=0)
        ys['ctr1'] = tf.dtypes.cast(ys['cls1'] == ys['cls2'], dtype=tf.float32)
        ys['ctr2'] = tf.dtypes.cast(ys['cls2'] == ys['cls3'], dtype=tf.float32)
            
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

def unet_block(input_layer, filters, kernel_size, strides):
    conv1 = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(input_layer)
    conv2 = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(conv1)
    return layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(conv2)

def create_autoencoder(input_layer, num_blocks, filter_scale, kernel_size, strides, max_pool):
    intermediate_layer = input_layer
    downsample = []
    for i in range(1, num_blocks+1):
        intermediate_layer = unet_block(intermediate_layer, filter_scale*(i), kernel_size, strides)
        downsample += [intermediate_layer]
        intermediate_layer = layers.MaxPool3D(pool_size=max_pool)(intermediate_layer)
    
    enc_layer = layers.Conv3D(filters=filter_scale*num_blocks*2, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(intermediate_layer)
    
    flatten_layer = layers.Flatten()(enc_layer)
    
    cls_layer = layers.Dense(1, activation="sigmoid")(flatten_layer)
    
    ctr_layer = layers.Dense(p['ctr_channels'], activation="sigmoid")(flatten_layer)
    
    intermediate_layer = enc_layer
    
    for i in range(1, num_blocks+1):
        intermediate_layer = layers.Conv3DTranspose(filters=filter_scale*((num_blocks+1)-i), kernel_size=kernel_size, strides=strides*2, activation="relu", padding="same")(intermediate_layer)
        intermediate_layer = layers.Concatenate()([intermediate_layer, downsample[num_blocks-i]])
        intermediate_layer = unet_block(intermediate_layer, filter_scale*((num_blocks+1)-i), kernel_size, strides)
    
    dec_layer = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=strides, activation="sigmoid", padding="same")(intermediate_layer)
    
    return [ctr_layer, cls_layer, dec_layer]


def prepare_model(inputs, use_cosine_similarity=True):
    inp = Input(INPUT_SHAPE)
    
    outputs = create_autoencoder(inp, p['num_blocks'], p['filter_scale'], p['kernel_size'], p['strides'], p['max_pool'])
    
    autoencoder_logits = {}
    autoencoder_logits["ctr"] = outputs[0]
    autoencoder_logits["cls"] = outputs[1]
    autoencoder_logits["dec"] = outputs[2]

    autoencoder_network = Model(inputs=inp, outputs=autoencoder_logits)

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
    siamese_logits["ctr1"] = layers.Layer(name="ctr1")(merge_layer1)
    siamese_logits["ctr2"] = layers.Layer(name="ctr2")(merge_layer2)
    siamese_logits["cls1"] = layers.Layer(name="cls1")(tower_1["cls"])
    siamese_logits["cls2"] = layers.Layer(name="cls2")(tower_2["cls"])
    siamese_logits["cls3"] = layers.Layer(name="cls3")(tower_3["cls"])
    siamese_logits["dec1"] = layers.Layer(name="dec1")(tower_1["dec"])
    siamese_logits["dec2"] = layers.Layer(name="dec2")(tower_2["dec"])
    siamese_logits["dec3"] = layers.Layer(name="dec3")(tower_3["dec"])
    
    siamese = Model(inputs=inputs, outputs=siamese_logits)
    
    siamese.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']),
        loss={
            'ctr1': contrastive_loss(),
            'ctr2': contrastive_loss(),
            'dec1': losses.MeanSquaredError(),
            'dec2': losses.MeanSquaredError(),
            'dec3': losses.MeanSquaredError(),
            'cls1': losses.BinaryCrossentropy(),
            'cls2': losses.BinaryCrossentropy(),
            'cls3': losses.BinaryCrossentropy()
        },
        loss_weights={
            'ctr1': p['ctr1'],
            'ctr2': p['ctr2'],
            'dec1': p['dec1'],
            'dec2': p['dec2'],
            'dec3': p['dec3'],
            'cls1': p['cls1'],
            'cls2': p['cls2'],
            'cls3': p['cls3']
        },
        metrics={
            'cls1': metrics.BinaryAccuracy(),
            'cls2': metrics.BinaryAccuracy(),
            'cls3': metrics.BinaryAccuracy()
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
steps_per_epoch = p['steps_per_epoch']
validation_freq = 1

# --- Determine total loop iterations needed
epochs = int(p['iterations'] / steps_per_epoch)

# --- Prepare Tensorboard 
log_dir = '{}/jmodels/logdirs/{}'.format(
    os.path.dirname(p['output_dir']),
    os.path.basename(p['output_dir']))


# --- Define training callbacks
tensorboard_log = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
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
    callbacks=[tensorboard_log, early_stopping]
)

# --- Save model
model.save(MODEL_NAME)
